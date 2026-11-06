import torch
import torch.nn.functional as F
from .geometry import placement


class H3WebUIFitImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'image': ('IMAGE',), 'width': ('INT', {'min': 32, 'max': 8192}),
            'height': ('INT', {'min': 32, 'max': 8192}), 'fit': (['pad', 'crop'],)}}

    RETURN_TYPES = ('IMAGE',)
    FUNCTION = 'fit'
    CATEGORY = 'MiniMax H3/WebUI'

    def fit(self, image, width, height, fit):
        w, h, x, y = placement(image.shape[2], image.shape[1], width, height, fit)
        parts = []
        for start in range(0, image.shape[0], 16):
            resized = F.interpolate(image[start:start + 16, :, :, :3].movedim(-1, 1),
                size=(h, w), mode='bilinear', align_corners=False, antialias=True)
            if fit == 'crop':
                resized = resized[:, :, -y:-y + height, -x:-x + width]
            else:
                resized = F.pad(resized, (x, width - w - x, y, height - h - y))
            parts.append(resized.movedim(1, -1))
        return (torch.cat(parts),)
