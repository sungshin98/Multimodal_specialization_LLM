"""KARINA 역할 2 통합 실행기.

예찬 파트의 EncoderOutput과 원본 파일 정보를 InputBridge로 묶어
Question Understanding Decoder -> Retrieval Gate -> Adaptive RAG를 실행한다.
기존 V4 핵심 구현은 유지하고, 역할 경계에서 필요한 입력 추적만 추가한다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .adaptive_multimodal_rag_V4 import AdaptiveMultimodalRAGPipelineV4
from .v4_contracts import (
    AdaptiveRAGOutput,
    InitialMultimodalContext,
    InputBridge,
    RoutedInput,
)


def _compat_from_encoder_outputs(
    cls,
    encoder_outputs: Optional[Any] = None,
    modality_summaries: Optional[Mapping[str, Any]] = None,
    metadata: Optional[Mapping[str, Mapping[str, Any]]] = None,
):
    """기존 V4.run()과 새 InputBridge 계약을 연결한다.

    V4가 처음 작성될 때 사용하던 from_encoder_outputs 호출을 유지하면서,
    이미 준비된 InitialMultimodalContext와 RoutedInput 목록도 그대로 받을 수 있게 한다.
    """

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


# 기존 AdaptiveMultimodalRAGPipelineV4를 깨지 않고 연결 계약을 보완한다.
if not hasattr(InitialMultimodalContext, "from_encoder_outputs"):
    InitialMultimodalContext.from_encoder_outputs = classmethod(  # type: ignore[attr-defined]
        _compat_from_encoder_outputs
    )


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
        """예찬 InputRouter를 파일별로 호출해 출처 관계를 보존하며 실행한다."""

        initial_context, _query_encoder_output = InputBridge.from_router_calls(
            router=router,
            query=question,
            file_paths=file_paths,
            content_hints=content_hints,
            source_metadata=source_metadata,
        )
        return self.run(
            question=question,
            initial_context=initial_context,
        )


__all__ = [
    "KARINARole2Pipeline",
    "InputBridge",
    "InitialMultimodalContext",
    "RoutedInput",
]
