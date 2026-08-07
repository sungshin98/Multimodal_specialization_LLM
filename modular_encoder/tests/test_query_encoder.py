from encoders.query.query_encoder import QueryEncoder

def main() -> None:
    output = QueryEncoder().encode("Who is the main character?")
    assert output.pooled_embedding.shape == (1, 384)
    assert output.modality == "query"
    print("QueryEncoder test passed:", output.pooled_embedding.shape)

if __name__ == "__main__":
    main()
