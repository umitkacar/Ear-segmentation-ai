from typing import List

from albumentations import Compose, Lambda, Resize


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        Resize(height=320, width=480, always_apply=True),
    ]
    return Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype("float32")


def get_preprocessing(preprocessing_fn) -> Compose:
    """Construct preprocessing transform
    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    """

    transform: List[Lambda] = [
        Lambda(image=preprocessing_fn),
        Lambda(image=to_tensor),
    ]
    return Compose(transform)
