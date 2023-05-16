import clip
import numpy as np
import torch
from PIL import Image

from encord_active.lib.embeddings.models.embedder_model import ImageEmbedder


class CLIPEmbedder(ImageEmbedder):
    def __init__(self):
        super().__init__(supports_text_embeddings=True)
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        tokens = torch.stack([clip.tokenize(t) for t in texts])
        return self.execute_with_largest_batch_size(self.model.encode_text, tokens)

    def embed_images(self, images: list[Image.Image]) -> np.ndarray:
        tensors = torch.stack([self.preprocess(i) for i in images])  # type: ignore
        return self.execute_with_largest_batch_size(self.model.encode_image, tensors)
