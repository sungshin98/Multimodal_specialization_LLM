"""KARINA 역할 2 통합 실행기.

예찬 파트의 EncoderOutput과 원본 파일 정보를 InputBridge로 묶어
Question Understanding Decoder -> Retrieval Gate -> Adaptive RAG를 실행한다.
기존 V4 핵심 구현은 유지하고, 역할 경계에서 필요한 입력 추적만 추가한다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .adaptive_multimodal_rag_V4 import (
    AdaptiveMultimodalRAGPipelineV4,
    PromptQuestionUnderstandingDecoder,
)
from .v4_contracts import (
    AdaptiveRAGOutput,
    InitialMultimodalContext,
    InputBridge,
    Modality,
    RoutedInput,
)


class SourceAwarePromptQuestionUnderstandingDecoder(PromptQuestionUnderstandingDecoder):
    """RoutedInput 계약에 맞춘 프롬프트 기반 Question Understanding Decoder."""

    SYSTEM_PROMPT = """
너는 KARINA의 질문 이해 디코더다. 답변을 생성하지 말고 원본 질문과
initial_inputs의 관계를 분석하여 구조화된 Question Condition을 생성하라.

initial_inputs의 각 항목
- source_id: 입력을 구분하는 식별자
- modality: text/image/video 등의 입력 종류
- source_path: 원본 파일 추적 정보
- content_hint: 입력의 의미를 설명하는 문서 내용, 캡션, OCR, 자막 또는 요약
- embedding_dim: 인코더 연결 검증 정보이며 입력의 의미 자체는 아니다

주의
- content_hint에 없는 이미지·영상 세부를 추측하지 마라.
- 검색 실행 여부를 직접 집행하지 않는다.
- external_knowledge_required와 검색용 sub_queries를 제시하고,
  최종 검색 결정은 Retrieval Gate가 수행한다.

반드시 JSON 객체만 반환한다.
필수 필드
- normalized_query: string
- input_interpretation: string
- task: string[]
- targets: string[]
- reasoning: string[]
- constraints: string[]
- required_modalities: string[]
- external_knowledge_required: boolean
- modality_focus: object
- sub_queries: [{query:string, modality:string, priority:low|medium|high}]
- answer_constraints: string[]
""".strip()


def _compat_from_encoder_outputs(
    cls,
    encoder_outputs: Optional[Any] = None,
    modality_summaries: Optional[Mapping[str, Any]] = None,
    metadata: Optional[Mapping[str, Mapping[str, Any]]] = None,
):
    """기존 V4.run()과 새 InputBridge 계약을 연결한다."""

    if isinstance(encoder_outputs, cls):
        return encoder_outputs

    if isinstance(encoder_outputs, Sequence) and not isinstance(
        encoder_outputs, (str, bytes)
    ):
        items = list(encoder_outputs)
        if items and all(isinstance(item, RoutedInput) for item in items):
            return cls(routed_inputs=items)

    return InputBridge.from_router_output(
        encoder_outputs=encoder_outputs,
        modality_summaries=modality_summaries,
        source_metadata=metadata,
    )


def _fixed_paths_by_modality(
    file_paths: Sequence[str | Path],
) -> Dict[str, List[Path]]:
    """동일 모달리티 파일이 여러 개일 때 경로가 중복 등록되지 않도록 한다."""

    grouped: Dict[str, List[Path]] = {}
    for value in file_paths:
        path = Path(value)
        extension = path.suffix.lower()
        if extension in {".txt", ".pdf", ".docx"}:
            modality = Modality.TEXT
            source_key = "document"
        elif extension in {".jpg", ".jpeg", ".png"}:
            modality = Modality.IMAGE
            source_key = modality.value
        elif extension in {".mp4", ".webm", ".mkv", ".avi", ".mov", ".ogv"}:
            modality = Modality.VIDEO
            source_key = modality.value
        else:
            continue

        grouped.setdefault(source_key, []).append(path)
        if source_key != modality.value:
            grouped.setdefault(modality.value, []).append(path)
    return grouped


# 기존 AdaptiveMultimodalRAGPipelineV4를 깨지 않고 새 연결 계약을 보완한다.
if not hasattr(InitialMultimodalContext, "from_encoder_outputs"):
    InitialMultimodalContext.from_encoder_outputs = classmethod(  # type: ignore[attr-defined]
        _compat_from_encoder_outputs
    )
InputBridge._paths_by_modality = staticmethod(_fixed_paths_by_modality)  # type: ignore[method-assign]


class KARINARole2Pipeline(AdaptiveMultimodalRAGPipelineV4):
    """역할 2의 권장 진입점.

    입력 우선순위
    1. initial_context
    2. routed_inputs
    3. encoder_outputs + file_paths + content_hints

    역할 3에는 반환값의 ``to_dict()`` 전체를 전달한다.
    """

    def run(
        self,
        question: str,
        *,
        initial_context: Optional[InitialMultimodalContext] = None,
        routed_inputs: Optional[Sequence[RoutedInput]] = None,
        encoder_outputs: Optional[Any] = None,
        file_paths: Optional[Sequence[str | Path]] = None,
        content_hints: Optional[Mapping[str, Any]] = None,
        source_metadata: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ) -> AdaptiveRAGOutput:
        if initial_context is not None and routed_inputs is not None:
            raise ValueError("initial_context와 routed_inputs는 동시에 지정할 수 없습니다.")

        if initial_context is None:
            if routed_inputs is not None:
                initial_context = InitialMultimodalContext(
                    routed_inputs=list(routed_inputs)
                )
            else:
                initial_context = InputBridge.from_router_output(
                    encoder_outputs=encoder_outputs,
                    file_paths=file_paths,
                    modality_summaries=content_hints,
                    source_metadata=source_metadata,
                )

        return super().run(
            question=question,
            encoder_outputs=initial_context,
        )

    def run_from_router(
        self,
        router: Any,
        question: str,
        file_paths: Sequence[str | Path],
        *,
        content_hints: Optional[Mapping[str, Any]] = None,
        source_metadata: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ) -> AdaptiveRAGOutput:
        """예찬 InputRouter의 원래 route() 계약을 사용해 바로 실행한다."""

        encoder_outputs = router.route(
            query=question,
            file_paths=list(file_paths),
        )
        initial_context = InputBridge.from_router_output(
            encoder_outputs=encoder_outputs,
            file_paths=file_paths,
            modality_summaries=content_hints,
            source_metadata=source_metadata,
        )
        return self.run(
            question=question,
            initial_context=initial_context,
        )


__all__ = [
    "KARINARole2Pipeline",
    "SourceAwarePromptQuestionUnderstandingDecoder",
    "InputBridge",
    "InitialMultimodalContext",
    "RoutedInput",
]
