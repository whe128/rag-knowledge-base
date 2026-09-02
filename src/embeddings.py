from langchain_huggingface import HuggingFaceEmbeddings
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["NO_PROXY"] = "*"

def get_embeddings(model_name = "BAAI/bge-m3"):
    """
        get embeddings from the specified model
    """

    return HuggingFaceEmbeddings(
        model_name = model_name,
        # parameter for the model inference
        # model_kwargs: initialization of the model with the specified parameters

        # device: "cpu" means the model will be loaded on the cpu
        # normalize_embeddings: True means the embeddings will be normalized to unit length
        encode_kwargs = {
            "device": "cpu",
            "normalize_embeddings": True # let the vector length be 1, the multiplication of normalized vectors is the similaraity of cosine, which is more accurate than the multiplication of unnormalized vectors
        }
    )
