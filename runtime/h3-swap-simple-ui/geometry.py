"""Shared preview/render geometry. No image is stretched to a new aspect ratio."""
import math

ASPECTS = {'16:9': (1536, 864), '9:16': (864, 1536), '1:1': (768, 768),
           '4:3': (1024, 768), '3:4': (768, 1024)}
RESOLUTIONS = {'384':384, '480':480, '576':576, '768':768, '1080':1080, '1440':1440}


def canvas(aspect, source=None, resolution=None):
    if resolution is not None:
        if resolution not in RESOLUTIONS:
            raise ValueError('未知分辨率档位')
        if aspect in ASPECTS:
            w,h = ASPECTS[aspect]
        elif aspect == 'auto':
            w,h = source or ASPECTS['16:9']
        else:
            raise ValueError('未知画幅比例')
        scale = RESOLUTIONS[resolution] / min(w,h)
        return max(32,round(w*scale/32)*32), max(32,round(h*scale/32)*32)
    if aspect in ASPECTS:
        return ASPECTS[aspect]
    if aspect != 'auto':
        raise ValueError('未知画幅比例')
    if not source:
        return 1344, 768
    w, h = source
    scale = min(768 / min(w, h), math.sqrt(1344 * 768 / (w * h)))
    return max(32, round(w * scale / 32) * 32), max(32, round(h * scale / 32) * 32)


def refined_size(w, h):
    # Keep an exact ratio on the32px grid when it still uses the2K pixel budget.
    divisor = math.gcd(w, h)
    unit_w, unit_h = 32 * w // divisor, 32 * h // divisor
    factor = int(min(2560 / max(unit_w, unit_h), math.sqrt(2560 * 1440 / (unit_w * unit_h))))
    if factor >= 1 and unit_w * unit_h * factor**2 >= 0.75 * 2560 * 1440:
        return unit_w * factor, unit_h * factor
    scale = min(2560 / max(w, h), math.sqrt(2560 * 1440 / (w * h)))
    return max(32, round(w * scale / 32) * 32), max(32, round(h * scale / 32) * 32)


def placement(sw, sh, width, height, fit):
    if fit not in ('crop', 'pad'):
        raise ValueError('请选择等比例裁剪或补边')
    scale = (max if fit == 'crop' else min)(width / sw, height / sh)
    w, h = round(sw * scale), round(sh * scale)
    if fit == 'crop':
        w, h = max(width, w), max(height, h)
    else:
        w, h = min(width, w), min(height, h)
    return w, h, (width - w) // 2, (height - h) // 2


def frames(seconds):
    seconds = float(seconds)
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('时长必须大于 0；推荐 5–15 秒')
    delivered = max(1, round(seconds * 24))
    internal = max(5, delivered)
    internal += (5 - internal) % 17
    return delivered, internal
