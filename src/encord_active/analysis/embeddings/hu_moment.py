import torch
from torch import FloatTensor, BoolTensor

from encord_active.analysis.embedding import PureObjectEmbedding


class HuMomentEmbeddings(PureObjectEmbedding):
    def __init__(self, ident: str) -> None:
        super().__init__(ident, set())

    def evaluate_embedding(self, mask: BoolTensor, coordinates: FloatTensor) -> FloatTensor:
        mask255 = mask.byte() * 255
        # moments = cv2.HuMoments(cv2.moments(mask)).flatten()
        return mask255.float()  # FIXME: return real result
