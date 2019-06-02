import os
from PIL import Image

from sites.bookcave import bookcave


def resize_image(path, size):
    path_parts = path.split(os.sep)
    base_ext = os.path.splitext(path_parts[-1])
    base, ext = base_ext[0], base_ext[1]
    out_fname = base + '-' + str(size[0]) + 'x' + str(size[1]) + ext
    if len(path_parts) > 1:
        out_path = os.sep.join(path_parts[:-1]) + os.sep + out_fname
    else:
        out_path = out_fname
    if os.path.exists(out_path):
        return
    try:
        image = Image.open(path)
        image_size = image.size
        image.thumbnail(size, Image.BICUBIC)
        out_image = Image.new("RGB", size)
        out_image.paste(image, ((size[0] - image.size[0]) // 2,
                                (size[1] - image.size[1]) // 2))
        out_image.save(out_path, 'JPEG')
        print('Resized `{}` from size {} to size {}.'.format(path, image_size, out_image.size))
    except IOError:
        print('Unable to resize `{}`.'.format(path))


def main():
    inputs, _, _, _ = bookcave.get_data({'images'}, images_source='cover')
    book_images = inputs['images']
    out_size = (512, 512)
    for images in book_images:
        path = images[0]
        resize_image(path, out_size)


if __name__ == '__main__':
    main()
