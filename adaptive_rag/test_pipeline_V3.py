from example_usage_V3 import pipeline
from adaptive_multimodal_rag_V3 import build_decoder_inputs


def test_multimodal_packet():
    output = pipeline.run(
        "포스터와 감독의 이전 작품을 비교하고 사실 여부를 검증해줘.",
        {"image": "어두운 색조의 영화 포스터"},
    )
    assert "text" in output.retrieval_results
    assert "image" in output.retrieval_results
    assert output.query_context["required_operations"] == ["compare", "verify"]
    decoder_inputs = build_decoder_inputs(output)
    assert "normalized_query" in decoder_inputs["text"]
    assert "focus_features" in decoder_inputs["image"]


if __name__ == "__main__":
    test_multimodal_packet()
    print("PASS")
