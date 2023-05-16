from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
import torch
from PIL import Image


class ImageEmbedder(ABC):
    def __init__(self, supports_text_embeddings: bool = False):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_batch_size: dict[Callable[[torch.Tensor], torch.Tensor], int] = {}
        self._supports_text_embeddings = supports_text_embeddings

    @torch.inference_mode()
    def execute_with_largest_batch_size(
        self, function: Callable[[torch.Tensor], torch.Tensor], input_: torch.Tensor
    ) -> np.ndarray:
        """
        Tries to utillize the GPU as much as possible by starting with the entire input.
        If it doesn't fit on the GPU, half the input is tried.
        If that doesn't fit halving again, and so on.
        The largest successful batch size will be remembered for successive calls to the function.

        Args:
            function: The function to be executed.
            input_: The tensor input.

        Returns:
            A numpy array with the result after concatenating the results of each batch.

        """
        if self.device == "cpu":
            return function(input_.to(self.device)).numpy()

        n, *_ = input_.shape
        if function in self.max_batch_size:
            bs = self.max_batch_size[function]
        else:
            bs = input_.shape[0]

        bs_updated = False
        while bs > 0:
            try:
                out = []
                for i in range(n // bs + 1):
                    start = i * bs
                    stop = (i + 1) * bs

                    if start >= n:
                        break

                    batch_out = function(input_[start:stop].to(self.device)).cpu()
                    out.append(batch_out)

                if bs_updated:
                    self.max_batch_size[function] = bs

                if len(out) == 1:
                    return out[0].numpy()

                return torch.concat(out, dim=0).numpy()

            except torch.cuda.OutOfMemoryError:  # type: ignore
                torch.cuda.empty_cache()
                bs = bs // 2
                bs_updated = True

        raise RuntimeError(
            'Not enough GPU memory to compute embeddings. Consider disabling GPU with the `CUDA_VISIBLE_DEVICES=""'
        )

    def _ensure_text_embeddings_enabled(self):
        if not self._supports_text_embeddings:
            raise RuntimeError("Embedder does not support text embeddings")

    def embed_text(self, text: str) -> np.ndarray:
        self._ensure_text_embeddings_enabled()
        return self.embed_texts([text]).squeeze()

    def embed_image(self, image: Image.Image):
        return self.embed_images([image]).squeeze()

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """
        If the `supports_text_embeddings` flag is enabled in the `__init__`
        command, provide a function for embedding text.

        Args:
            texts: The list of length N of text that should be embedded.

        Returns:
            A numpy array of shape [N, -1] with the result after concatenating
            the results of each text.

        """
        self._ensure_text_embeddings_enabled()
        return np.empty((0, 512))

    @abstractmethod
    def embed_images(self, images: list[Image.Image]) -> np.ndarray:
        """
        The function for embeddings images. Note that you can utililze the
        `self.execute_with_largest_batch_size` function to utilize the GPU
        optimally.

        Args:
            images: The list (of length N) of images that should be embedded.

        Returns:
            A numpy array of shape [N, -1] with the resulting embeddings after
            concatenating the results of each image.
        """
        ...
