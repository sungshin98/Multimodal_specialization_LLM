from sentence_transformers import SentenceTransformer

from interfaces.encoder_output import EncoderOutput


class QueryEncoder:
    def __init__(
        self,
        model_name: str = (
            "sentence-transformers/"
            "paraphrase-multilingual-MiniLM-L12-v2"
        ),
    ) -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, query: str) -> EncoderOutput:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")

        pooled_embedding = self.model.encode(
            [query],
            convert_to_tensor=True,
            normalize_embeddings=False,
        )

        return EncoderOutput(
            pooled_embedding=pooled_embedding,
            modality="query",
        )
