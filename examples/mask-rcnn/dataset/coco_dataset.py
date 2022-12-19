from typing import Any, Callable, Optional, Tuple

from torchvision.datasets import CocoDetection


class EncordCocoDetection(CocoDetection):
    """
    Adaptation of TorchVision CocoDetection to pass image metadata as part of iterator.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.

    Returns:
        tuple: Tuple (image, target, metadata) where target is the object returned by ``coco.loadAnns`` and metadata is an EncordCocoMetadata object.
    """

    def __init__(
        self,
        root: str,
        annFile: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, annFile, transform, target_transform, transforms)

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)
        metadata = self.coco.loadImgs(id)[0]

        label_hash, _, data_hash_ext = metadata["file_name"].split("/")[-3:]
        data_hash = data_hash_ext.split(".")[0]
        metadata["label_hash"] = label_hash
        metadata["data_hash"] = data_hash

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target, metadata

    @property
    def num_classes(self) -> int:
        return len(self.coco.cats)
