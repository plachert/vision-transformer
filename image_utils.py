import numpy as np


def convert_to_255scale(image):
    """Convert an image to 255 scale."""
    clipped = np.clip(image, 0., 1.)
    image_255 = 255 * clipped
    return image_255.astype(np.uint8)


def channel_last(image):
    """Channel first to channel last."""
    transposed = np.transpose(image, (1, 2, 0))
    return transposed


def channel_first(image):
    """Channel last to channel first."""
    transposed = np.transpose(image, (2, 0, 1))
    return transposed


if __name__ == "__main__":
    image = np.random((224, 224, 3), dtype=np.uint8)
    print(image.shape)