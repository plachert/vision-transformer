import torch


class Patchify:
    def __init__(self, patch_size=(16, 16), flatten_patch_dims=True):
        self.patch_h, self.patch_w = patch_size
        self.flatten_patch_dims = flatten_patch_dims
        
    def validate_input(self, image):
        """Check if the images shape is compatible with selected patch_size."""
        _, _, h, w = image.shape
        assert h % self.patch_h == 0
        assert w % self.patch_w == 0
        return True
        
    def __call__(self, image: torch.Tensor):
        c, h, w = image.shape
        n_rows = h // self.patch_h
        n_columns =  w // self.patch_w
        patches = image.reshape(c, n_rows, self.patch_h, n_columns, self.patch_w)
        patches = patches.permute(1, 3, 0, 2, 4).flatten(0, 1) # n, n_patches, c, patch_h, patch_w
        if self.flatten_patch_dims:
            patches = patches.flatten(1, 3) # n, n_patches, c * patch_h * patch_w (embedding)
        return patches
        
if __name__ == "__main__":
    data = torch.rand((3, 224, 224))
    patchify = Patchify(flatten_patch_dims=False)
    patchify_flat = Patchify(flatten_patch_dims=True)
    print(patchify(data).shape)
    print(patchify_flat(data).shape)
    
        

