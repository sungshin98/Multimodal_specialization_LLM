from encoders.document.document_encoder import DocumentEncoder

def main() -> None:
    output = DocumentEncoder().encode("A detective investigates a case.")
    assert output.pooled_embedding.shape == (1, 384)
    assert output.modality == "document"
    print("DocumentEncoder test passed:", output.pooled_embedding.shape)

if __name__ == "__main__":
    main()
