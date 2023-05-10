from typing import Callable

import clip
import numpy as np
import torch
import torch.cuda
from PIL import Image


class CLIPEmbedder:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.max_batch_size: dict[Callable[[torch.Tensor], torch.Tensor], int] = {}

    @torch.inference_mode()
    def execute_with_largest_batch_size(
        self, function: Callable[[torch.Tensor], torch.Tensor], input_: torch.Tensor
    ) -> np.ndarray:
        """
        Tries to utillize the GPU as much as possible by starting with the entire input.
        If it doesn't fit on the GPU, half the input is tried.
        If that doesn't fit halving again, and so on.
        The largest successfull batch size will be remembered for successive calls to the function.

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

    def embed_text(self, text: str) -> np.ndarray:
        return self.embed_texts([text]).squeeze()

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        tokens = torch.stack([clip.tokenize(t) for t in texts])
        return self.execute_with_largest_batch_size(self.model.encode_text, tokens)

    def embed_image(self, image: Image.Image):
        return self.embed_images([image]).squeeze()

    def embed_images(self, images: list[Image.Image]) -> np.ndarray:
        tensors = torch.stack([self.preprocess(i) for i in images])  # type: ignore
        return self.execute_with_largest_batch_size(self.model.encode_image, tensors)
