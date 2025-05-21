from PIL import Image, ImageOps
import os


def generate_image_version(input_image, output_dir, threshold=128):
    os.makedirs(output_dir, exist_ok=True)
    img = Image.open(input_image)
    image_name = input_image.split('\\')[-1].split('.')[0]
    print(image_name)

    """Convert to grayscale"""
    gray_img = img.convert('L')
    gray_path = os.path.join(output_dir, f'{image_name}_grayscale.png')
    gray_img.save(gray_path)

    """Convert to dither image"""
    dither_img = img.convert('1')  # Автоматически применяет дизеринг Флойда-Стейнберга
    dither_path = os.path.join(output_dir, f'{image_name}_bw_dither.png')
    dither_img.save(dither_path)

    """Convert to bw image"""
    bw_img = img.convert('L')
    bw_img = bw_img.point(lambda x: 255 if x > threshold else 0, '1')
    bw_path = os.path.join(output_dir, f'{image_name}_bw_threshold.png')
    bw_img.save(bw_path)


input_image = r'huge_image_2048_2048.jpg'
dir_name = 'Converted_images'

generate_image_version(input_image, dir_name)
