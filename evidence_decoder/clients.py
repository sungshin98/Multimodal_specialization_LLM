"""LLM / VLM 클라이언트.

외부 SDK 의존 없이 urllib 만 사용한다. 실험 환경 재현이 목적이라
pip 설치 실패로 파이프라인이 멈추는 상황을 만들지 않는다.

모달리티별로 다른 클라이언트를 주입하는 구조다.
- 텍스트 / 통합 / 최종 : SolarStructuredClient (Upstage solar-pro3)
- 이미지 / 영상       : GeminiVisionClient 또는 OpenAIVisionClient
                        키가 없으면 CaptionFallbackVisionClient
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence

UPSTAGE_BASE_URL = "https://api.upstage.ai/v1"
OPENAI_BASE_URL = "https://api.openai.com/v1"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

# 실측 기준 (2026-08, 근거카드 생성 프롬프트):
#   solar-mini 5.89s / solar-pro2 1.75s / solar-pro3 1.30s
#   solar-mini 는 카드 개수 지시를 어겨 탈락.
DEFAULT_SOLAR_MODEL = "solar-pro3"
# 실측 기준 (2026-08, 399KB 이미지 + JSON 스키마 강제):
#   gemini-3.5-flash 5.09s / gemini-3.6-flash 6.37s / gemini-flash-latest 6.80s
#   gemini-2.5-flash 는 신규 사용자에게 더 이상 제공되지 않는다(404).
#   -latest 별칭은 모델이 바뀔 수 있어 실험 재현성을 위해 고정 이름을 쓴다.
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"
DEFAULT_OPENAI_VISION_MODEL = "gpt-4.1-mini"


class LLMError(RuntimeError):
    pass


@dataclass
class MediaAsset:
    """비전 모델에 넘길 이미지/영상 1건."""

    data: bytes
    mime_type: str
    label: str = ""
    text_hint: str = ""

    def to_data_uri(self) -> str:
        return f"data:{self.mime_type};base64,{base64.b64encode(self.data).decode()}"

    @classmethod
    def from_path(cls, path: str, label: str = "", text_hint: str = "") -> "MediaAsset":
        mime, _ = mimetypes.guess_type(path)
        with open(path, "rb") as handle:
            return cls(
                data=handle.read(),
                mime_type=mime or "application/octet-stream",
                label=label or os.path.basename(path),
                text_hint=text_hint,
            )


class StructuredLLMClient(Protocol):
    """adaptive_rag 쪽 StructuredLLMClient 와 호환되는 인터페이스."""

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: Optional[Mapping[str, Any]] = None,
    ) -> Mapping[str, Any]:
        ...


class VisionStructuredClient(Protocol):
    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        assets: Sequence[MediaAsset] = (),
        schema: Optional[Mapping[str, Any]] = None,
    ) -> Mapping[str, Any]:
        ...


# ============================================================
# 공통 HTTP
# ============================================================


@dataclass
class CallStats:
    calls: int = 0
    total_ms: float = 0.0
    retries: int = 0

    def record(self, elapsed_ms: float, retries: int = 0) -> None:
        self.calls += 1
        self.total_ms += elapsed_ms
        self.retries += retries


def _post_json(
    url: str,
    payload: Mapping[str, Any],
    headers: Mapping[str, str],
    timeout: float,
) -> Dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")[:400]
        raise LLMError(f"HTTP {error.code}: {detail}") from error
    except urllib.error.URLError as error:
        raise LLMError(f"연결 실패: {error.reason}") from error


def _loads_lenient(text: str) -> Dict[str, Any]:
    """json_schema strict 를 쓰면 거의 필요 없지만, 폴백 모델용 방어."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.split("```")[1] if len(text.split("```")) > 1 else text
        if text.lstrip().lower().startswith("json"):
            text = text.lstrip()[4:]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
    raise LLMError(f"JSON 파싱 실패: {text[:200]}")


# ============================================================
# Upstage Solar - 텍스트 축 전담
# ============================================================


class SolarStructuredClient:
    """Upstage Solar. json_schema strict 모드로 스키마를 강제한다.

    strict 모드가 거부되면 json_object 로 1회 폴백한다.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_SOLAR_MODEL,
        temperature: float = 0.0,
        timeout: float = 90.0,
        max_retries: int = 2,
        base_url: str = UPSTAGE_BASE_URL,
    ) -> None:
        key = api_key or os.getenv("UPSTAGE_API_KEY")
        if not key:
            raise LLMError("UPSTAGE_API_KEY 가 없다. .env 를 확인할 것.")
        self.api_key = key
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self.base_url = base_url.rstrip("/")
        self.stats = CallStats()
        self._strict_supported = True

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: Optional[Mapping[str, Any]] = None,
    ) -> Mapping[str, Any]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        started = time.perf_counter()
        retries = 0
        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            payload: Dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "response_format": self._response_format(schema),
            }
            try:
                data = _post_json(
                    f"{self.base_url}/chat/completions",
                    payload,
                    {"Authorization": f"Bearer {self.api_key}"},
                    self.timeout,
                )
                content = data["choices"][0]["message"]["content"]
                self.stats.record((time.perf_counter() - started) * 1000, retries)
                return _loads_lenient(content)
            except LLMError as error:
                last_error = error
                retries += 1
                if schema is not None and self._strict_supported and "json_schema" in str(error):
                    self._strict_supported = False  # 서버가 strict 를 거부 -> json_object 로
                    continue
                if attempt < self.max_retries:
                    time.sleep(0.5 * (attempt + 1))

        self.stats.record((time.perf_counter() - started) * 1000, retries)
        raise LLMError(f"{self.model} 호출 실패: {last_error}")

    def _response_format(self, schema: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        if schema is not None and self._strict_supported:
            return {
                "type": "json_schema",
                "json_schema": {"name": "decoder_output", "strict": True, "schema": dict(schema)},
            }
        return {"type": "json_object"}


# ============================================================
# 비전 클라이언트
# ============================================================


class GeminiVisionClient:
    """Gemini Flash. 영상을 프레임 분해 없이 그대로 받는다.

    이미지/영상 디코더의 기본 선택지. GOOGLE_API_KEY 필요.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_GEMINI_MODEL,
        temperature: float = 0.0,
        timeout: float = 180.0,
        max_retries: int = 2,
        thinking: Optional[str] = "low",
    ) -> None:
        key = api_key or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise LLMError("GOOGLE_API_KEY 가 없다.")
        self.api_key = key
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        # Gemini 3.x 는 내부 추론이 기본으로 켜져 있다. 실측으로 이미지 443,
        # 영상 435 사고 토큰을 소비했고 이것이 지연에 직접 반영된다.
        #   이미지 4.99s(기본) -> 4.32s(budget=0) -> 3.97s(level=low)
        # 다만 끄면 서술 정확도가 떨어진다. 같은 영상을 두고 기본은
        # "빽빽하게 깔린 넙치 무리", 끔은 "흩어져 있다"로 상반된 판독을 냈다.
        # 그래서 완전히 끄지 않고 'low' 를 기본값으로 둔다.
        # None 이면 모델 기본값을 쓴다.
        self.thinking = thinking
        self.stats = CallStats()

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        assets: Sequence[MediaAsset] = (),
        schema: Optional[Mapping[str, Any]] = None,
    ) -> Mapping[str, Any]:
        parts: List[Dict[str, Any]] = [{"text": user_prompt}]
        for asset in assets:
            if asset.label:
                parts.append({"text": f"[자료: {asset.label}]"})
            parts.append(
                {
                    "inline_data": {
                        "mime_type": asset.mime_type,
                        "data": base64.b64encode(asset.data).decode(),
                    }
                }
            )

        generation_config: Dict[str, Any] = {
            "temperature": self.temperature,
            "response_mime_type": "application/json",
        }
        if schema is not None:
            generation_config["response_schema"] = _to_gemini_schema(schema)
        if self.thinking is not None:
            generation_config["thinkingConfig"] = (
                {"thinkingBudget": self.thinking}
                if isinstance(self.thinking, int)
                else {"thinkingLevel": self.thinking}
            )

        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": generation_config,
        }
        url = f"{GEMINI_BASE_URL}/models/{self.model}:generateContent"

        started = time.perf_counter()
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                data = _post_json(url, payload, {"x-goog-api-key": self.api_key}, self.timeout)
                text = data["candidates"][0]["content"]["parts"][0]["text"]
                self.stats.record((time.perf_counter() - started) * 1000, attempt)
                return _loads_lenient(text)
            except (LLMError, KeyError, IndexError) as error:
                last_error = error
                if attempt < self.max_retries:
                    time.sleep(0.5 * (attempt + 1))

        self.stats.record((time.perf_counter() - started) * 1000, self.max_retries)
        raise LLMError(f"{self.model} 호출 실패: {last_error}")


class OpenAIVisionClient:
    """OpenAI 호환 비전 엔드포인트. 영상은 샘플 프레임을 이미지로 넣는다."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_OPENAI_VISION_MODEL,
        temperature: float = 0.0,
        timeout: float = 180.0,
        max_retries: int = 2,
        base_url: str = OPENAI_BASE_URL,
    ) -> None:
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise LLMError("OPENAI_API_KEY 가 없다.")
        self.api_key = key
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self.base_url = base_url.rstrip("/")
        self.stats = CallStats()

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        assets: Sequence[MediaAsset] = (),
        schema: Optional[Mapping[str, Any]] = None,
    ) -> Mapping[str, Any]:
        content: List[Dict[str, Any]] = [{"type": "text", "text": user_prompt}]
        for asset in assets:
            if asset.label:
                content.append({"type": "text", "text": f"[자료: {asset.label}]"})
            content.append({"type": "image_url", "image_url": {"url": asset.to_data_uri()}})

        response_format: Dict[str, Any] = {"type": "json_object"}
        if schema is not None:
            response_format = {
                "type": "json_schema",
                "json_schema": {"name": "decoder_output", "strict": True, "schema": dict(schema)},
            }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            "temperature": self.temperature,
            "response_format": response_format,
        }

        started = time.perf_counter()
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                data = _post_json(
                    f"{self.base_url}/chat/completions",
                    payload,
                    {"Authorization": f"Bearer {self.api_key}"},
                    self.timeout,
                )
                self.stats.record((time.perf_counter() - started) * 1000, attempt)
                return _loads_lenient(data["choices"][0]["message"]["content"])
            except (LLMError, KeyError, IndexError) as error:
                last_error = error
                if attempt < self.max_retries:
                    time.sleep(0.5 * (attempt + 1))

        self.stats.record((time.perf_counter() - started) * 1000, self.max_retries)
        raise LLMError(f"{self.model} 호출 실패: {last_error}")


class ResilientVisionClient:
    """비전 백엔드가 런타임에 죽어도 이미지/영상 모달리티를 통째로 잃지 않는다.

    키 만료(401), 쿼터 초과(429), 모델 미지원 같은 사고는 프로세스 시작 시점에
    알 수 없다. 첫 호출에서 실패하면 폴백으로 내려가고, 이후 호출은 폴백을 바로 쓴다.
    """

    def __init__(self, primary: VisionStructuredClient, fallback: VisionStructuredClient) -> None:
        self.primary = primary
        self.fallback = fallback
        self.degraded = False
        self.failure_reason = ""

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        assets: Sequence[MediaAsset] = (),
        schema: Optional[Mapping[str, Any]] = None,
    ) -> Mapping[str, Any]:
        if not self.degraded:
            try:
                return self.primary.generate_json(system_prompt, user_prompt, assets, schema)
            except LLMError as error:
                self.degraded = True
                self.failure_reason = str(error)[:200]
        return self.fallback.generate_json(system_prompt, user_prompt, assets, schema)


class CaptionFallbackVisionClient:
    """비전 모델 키가 없을 때 쓰는 대체 경로.

    원본 픽셀 대신 metadata 의 캡션/OCR/자막 텍스트만 텍스트 LLM 에 넘긴다.
    파이프라인이 멈추지 않게 하는 것이 목적이며, 실험 결과에는
    degraded=True 로 표시되어 정상 경로와 구분된다.
    """

    degraded = True

    def __init__(self, text_client: StructuredLLMClient) -> None:
        self.text_client = text_client

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        assets: Sequence[MediaAsset] = (),
        schema: Optional[Mapping[str, Any]] = None,
    ) -> Mapping[str, Any]:
        hints = []
        for asset in assets:
            hint = asset.text_hint.strip()
            hints.append(
                f"[자료: {asset.label or '이름없음'}]\n{hint if hint else '(설명 텍스트 없음 - 파일명만 확인 가능)'}"
            )
        note = (
            "\n\n[중요] 비전 모델을 사용할 수 없어 원본 이미지/영상을 직접 보지 못했다. "
            "아래 텍스트 설명만으로 판단하고, 설명이 부족한 근거는 confidence 를 0.3 이하로 낮추고 "
            "추측으로 시각적 세부를 지어내지 마라.\n" + "\n\n".join(hints)
        )
        return self.text_client.generate_json(system_prompt, user_prompt + note, schema)


# ============================================================
# 오프라인 테스트용
# ============================================================


@dataclass
class ScriptedClient:
    """네트워크 없이 파이프라인 구조를 테스트하기 위한 결정적 클라이언트."""

    responses: List[Mapping[str, Any]] = field(default_factory=list)
    default: Mapping[str, Any] = field(default_factory=dict)
    stats: CallStats = field(default_factory=CallStats)
    _index: int = 0

    def generate_json(self, system_prompt, user_prompt, assets=(), schema=None):  # noqa: D102
        self.stats.record(0.0)
        if self._index < len(self.responses):
            response = self.responses[self._index]
            self._index += 1
            return response
        return self.default


def _to_gemini_schema(schema: Mapping[str, Any]) -> Dict[str, Any]:
    """OpenAI JSON Schema -> Gemini response_schema.

    Gemini 는 additionalProperties 를 받지 않고 타입명을 대문자로 쓴다.
    """
    if not isinstance(schema, Mapping):
        return {}

    converted: Dict[str, Any] = {}
    for key, value in schema.items():
        if key == "additionalProperties":
            continue
        if key == "type" and isinstance(value, str):
            converted["type"] = value.upper()
        elif key == "properties" and isinstance(value, Mapping):
            converted["properties"] = {k: _to_gemini_schema(v) for k, v in value.items()}
        elif key == "items" and isinstance(value, Mapping):
            converted["items"] = _to_gemini_schema(value)
        else:
            converted[key] = value
    return converted


def build_default_clients(
    vision_preference: Sequence[str] = ("gemini", "openai"),
) -> Dict[str, Any]:
    """환경변수를 보고 사용 가능한 클라이언트를 조립한다.

    반환: {"text": 텍스트클라이언트, "vision": 비전클라이언트, "vision_backend": 이름}
    """
    text_client = SolarStructuredClient()
    fallback = CaptionFallbackVisionClient(text_client)

    for backend in vision_preference:
        try:
            if backend == "gemini":
                primary: VisionStructuredClient = GeminiVisionClient()
            elif backend == "openai":
                primary = OpenAIVisionClient()
            else:
                continue
        except LLMError:
            continue  # 키 없음 - 다음 후보로

        # 키가 있어도 만료/쿼터 문제로 런타임에 죽을 수 있다. 폴백을 붙여 둔다.
        return {
            "text": text_client,
            "vision": ResilientVisionClient(primary, fallback),
            "vision_backend": backend,
        }

    return {"text": text_client, "vision": fallback, "vision_backend": "caption_fallback"}
