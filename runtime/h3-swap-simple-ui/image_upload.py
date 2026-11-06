from PIL import Image


IMAGE_FORMATS = {'JPEG': '.jpg', 'PNG': '.png', 'WEBP': '.webp',
                 'BMP': '.bmp', 'TIFF': '.tiff', 'GIF': '.gif'}
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff', '.gif'}


def image_suffix(path):
    with Image.open(path) as image:
        suffix = IMAGE_FORMATS.get(image.format)
        if not suffix:
            raise ValueError('支持 JPG/JPEG、PNG、WebP、BMP、TIFF 和 GIF 图片')
        image.verify()
    with Image.open(path) as image:
        image.load()
    return suffix
