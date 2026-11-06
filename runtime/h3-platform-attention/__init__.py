"""Platform attention shim.

gfx1151 keeps using the separate ROCm CK node when that package is installed.
Every other platform, including NVIDIA Thor, registers a no-op so the existing
H3 graph runs with PyTorch attention. No weights live in this package.
"""


class H3NativeAttention:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'model': ('MODEL',)}}

    RETURN_TYPES = ('MODEL',)
    FUNCTION = 'patch'
    CATEGORY = 'MiniMax H3/attention'

    def patch(self, model):
        return (model,)


NODE_CLASS_MAPPINGS = {'H3NativeAttention': H3NativeAttention}
NODE_DISPLAY_NAME_MAPPINGS = {'H3NativeAttention': 'H3 PyTorch Attention'}
