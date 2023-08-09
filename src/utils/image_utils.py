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


def image_to_patches(image, patch_size=(16, 16)):
    """Divide image into patches."""
    h, w, c = image.shape
    assert w % patch_size[0] == 0
    assert h % patch_size[1] == 0
    n_columns =  w // patch_size[0]
    n_rows = h // patch_size[1]
    patches = image.reshape(n_columns, patch_size[0], n_rows, patch_size[1], c)
    patches = patches.transpose(0, 2, 1, 3, 4).reshape(-1, patch_size[0], patch_size[1], c) # Perhaps, this can be improved
    return patches
    
    

if __name__ == "__main__":
    image = np.random.rand(224, 224, 3)
    print(image.shape)
    patches = image_to_patches(image).shape