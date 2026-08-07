from sentence_transformers import SentenceTransformer

from interfaces.encoder_output import EncoderOutput


class DocumentEncoder:
    def __init__(
        self,
        model_name: str = (
            "sentence-transformers/"
            "paraphrase-multilingual-MiniLM-L12-v2"
        ),
    ) -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, document: str) -> EncoderOutput:
        if not isinstance(document, str) or not document.strip():
            raise ValueError(
                "document must be a non-empty string."
            )

        pooled_embedding = self.model.encode(
            [document],
            convert_to_tensor=True,
            normalize_embeddings=False,
        )

        return EncoderOutput(
            pooled_embedding=pooled_embedding,
            modality="document",
        )
