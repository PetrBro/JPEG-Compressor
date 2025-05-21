import numpy as np
from PIL import Image
from scipy.fftpack import dct, idct
import os
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt

# Константы
BLOCK_SIZE = 8
DEFAULT_QUALITY = 50

# Стандартные матрицы квантования (Luma и Chroma)
Q_LUMA = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
], dtype=np.int32)

Q_CHROMA = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]
], dtype=np.int32)

# Zig-zag порядок
ZIGZAG_ORDER = [
    (0, 0), (0, 1), (1, 0), (2, 0), (1, 1), (0, 2), (0, 3), (1, 2),
    (2, 1), (3, 0), (4, 0), (3, 1), (2, 2), (1, 3), (0, 4), (0, 5),
    (1, 4), (2, 3), (3, 2), (4, 1), (5, 0), (6, 0), (5, 1), (4, 2),
    (3, 3), (2, 4), (1, 5), (0, 6), (0, 7), (1, 6), (2, 5), (3, 4),
    (4, 3), (5, 2), (6, 1), (7, 0), (7, 1), (6, 2), (5, 3), (4, 4),
    (3, 5), (2, 6), (1, 7), (2, 7), (3, 6), (4, 5), (5, 4), (6, 3),
    (7, 2), (7, 3), (6, 4), (5, 5), (4, 6), (3, 7), (4, 7), (5, 6),
    (6, 5), (7, 4), (7, 5), (6, 6), (5, 7), (6, 7), (7, 6), (7, 7)
]

# Кодирование Хаффмана (упрощенные таблицы)
HUFFMAN_DC_LUMA = {
    0: '00', 1: '010', 2: '011', 3: '100',
    4: '101', 5: '110', 6: '1110', 7: '11110',
    8: '111110', 9: '1111110', 10: '11111110', 11: '111111110'
}

HUFFMAN_AC_LUMA = {
    (0, 0): '1010', (0, 1): '00', (0, 2): '01', (0, 3): '100',
    (0, 4): '1011', (0, 5): '11010', (0, 6): '111000',
    (0, 7): '1111000', (0, 8): '1111110110', (0, 9): '1111111110000010',
    (0, 10): '1111111110000011', (1, 1): '1100', (1, 2): '11011',
    (1, 3): '1111001', (1, 4): '111110110', (1, 5): '11111110110',
    # ... (дополнительные коды)
}


# Вспомогательные функции
def rgb_to_ycbcr(rgb_image):
    ycbcr = rgb_image.convert('YCbCr')
    y, cb, cr = ycbcr.split()
    return np.array(y), np.array(cb), np.array(cr)


def downsample(channel, factor):
    h, w = channel.shape
    return channel[::factor, ::factor]


def pad_image(image, block_size):
    h, w = image.shape
    pad_h = (block_size - h % block_size) % block_size
    pad_w = (block_size - w % block_size) % block_size
    return np.pad(image, ((0, pad_h), (0, pad_w))), (h, w)


def split_blocks(image, block_size):
    h, w = image.shape
    return [image[i:i + block_size, j:j + block_size]
            for i in range(0, h, block_size)
            for j in range(0, w, block_size)]


def apply_dct(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')


def quantize(block, q_table):
    return np.round(block / q_table).astype(np.int32).tolist()


def dequantize(block, q_table):
    return block * q_table


def inverse_dct(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')


def zigzag(block):
    """Принимает 2D блок 8x8 (numpy array или список списков)"""
    return [block[coord[0]][coord[1]] if isinstance(block, list) else block[coord]
            for coord in ZIGZAG_ORDER]


def unzigzag(zigzag_list):
    block = np.zeros((BLOCK_SIZE, BLOCK_SIZE))
    for i, coord in enumerate(ZIGZAG_ORDER[1:]):  # Пропускаем DC компонент
        if i < len(zigzag_list):
            block[coord] = zigzag_list[i]
    return block


def rle_encode(ac_coefficients):
    rle = []
    zero_count = 0
    for coeff in ac_coefficients:
        if coeff == 0:
            zero_count += 1
        else:
            rle.append((zero_count, coeff))
            zero_count = 0
    if zero_count > 0:
        rle.append((0, 0))  # EOB
    return rle


def rle_decode(rle_list):
    ac = []
    for run, value in rle_list:
        if run == 0 and value == 0:  # EOB
            remaining = 63 - len(ac)
            ac.extend([0] * remaining)
            break
        ac.extend([0] * run)
        ac.append(value)
    # Гарантированно возвращаем 63 элемента
    return ac[:63] + [0]*(63 - len(ac[:63]))

def upsample(channel, factor):
    h, w = channel.shape
    return np.repeat(np.repeat(channel, factor, axis=0), factor, axis=1)


# Основной класс компрессора
class JPEGCompressor:
    def __init__(self, quality=DEFAULT_QUALITY, mode='compress'):
        self.quality = quality
        self.mode = mode
        self.q_luma = self.scale_quantization_table(Q_LUMA)
        self.q_chroma = self.scale_quantization_table(Q_CHROMA)

    def scale_quantization_table(self, table):
        if self.mode == 'plot':
            effective_quality = 100 - self.quality
            if effective_quality <= 0:
                scale = 5000 / 1
            elif effective_quality < 50:
                scale = 5000 / effective_quality
            else:
                scale = 200 - 2 * effective_quality
        else:
            scale = self.quality * 100/4
        scaled_table = np.clip(np.floor((table * scale + 50) / 100), 1, 255)
        return scaled_table.astype(np.int32)

    def compress(self, input_path, output_path):
        # Загрузка и конвертация
        img = Image.open(input_path)
        y, cb, cr = rgb_to_ycbcr(img)

        # Даунсэмплинг
        cb_down = downsample(cb, 2)
        cr_down = downsample(cr, 2)

        # Подготовка каналов
        y_padded, y_shape = pad_image(y, BLOCK_SIZE)
        cb_padded, cb_shape = pad_image(cb_down, BLOCK_SIZE)
        cr_padded, cr_shape = pad_image(cr_down, BLOCK_SIZE)

        # Разбиение на блоки
        y_blocks = split_blocks(y_padded, BLOCK_SIZE)
        cb_blocks = split_blocks(cb_padded, BLOCK_SIZE)
        cr_blocks = split_blocks(cr_padded, BLOCK_SIZE)

        # Обработка блоков
        compressed_data = {
            'metadata': {
                'y_shape': y_shape,
                'cb_shape': cb_shape,
                'cr_shape': cr_shape,
                'quality': self.quality
            },
            'q_luma': self.q_luma,
            'q_chroma': self.q_chroma,
            'blocks': []
        }

        for channel, q_table in zip([y_blocks, cb_blocks, cr_blocks],
                                    [self.q_luma, self.q_chroma, self.q_chroma]):
            dc_prev = 0
            channel_blocks = []
            for block in channel:
                # DCT
                dct_block = apply_dct(block)
                # Квантование
                quantized = quantize(dct_block, q_table)
                # DC разность
                dc = quantized[0][0]
                dc_diff = dc - dc_prev
                dc_prev = dc

                zigzag_full = zigzag(quantized)
                ac_zigzag = zigzag_full[1:]
                ac_rle = rle_encode(ac_zigzag)

                channel_blocks.append({
                    'dc_diff': dc_diff,
                    'ac_rle': ac_rle
                })
            compressed_data['blocks'].append(channel_blocks)


        with open(output_path, 'wb') as f:
            pickle.dump(compressed_data, f)


class JPEGDecompressor:
    def __init__(self):
        self.inverse_huffman_dc = {v: k for k, v in HUFFMAN_DC_LUMA.items()}
        self.inverse_huffman_ac = {v: k for k, v in HUFFMAN_AC_LUMA.items()}

    def decode_dc(self, reader):
        bits = ''
        while True:
            bit = reader.read_bit()
            if bit is None: break
            bits += str(bit)
            if bits in self.inverse_huffman_dc:
                size = self.inverse_huffman_dc[bits]
                break
        if size == 0: return 0
        value_bits = reader.read_bits(size)
        value = int(value_bits, 2)
        if value >= (1 << (size - 1)):
            value -= (1 << size)
        return value

    def decode_ac(self, reader):
        bits = ''
        while True:
            bit = reader.read_bit()
            if bit is None: return (0, 0)
            bits += str(bit)
            if bits in self.inverse_huffman_ac:
                run, size = self.inverse_huffman_ac[bits]
                break
        if size == 0: return (run, 0)
        value_bits = reader.read_bits(size)
        value = int(value_bits, 2)
        if value >= (1 << (size - 1)):
            value -= (1 << size)
        return (run, value)

    def decompress(self, input_path, output_path):
        import pickle
        with open(input_path, 'rb') as f:
            compressed_data = pickle.load(f)

        metadata = compressed_data['metadata']
        q_luma = compressed_data['q_luma']
        q_chroma = compressed_data['q_chroma']
        blocks_data = compressed_data['blocks']

        y_h, y_w = metadata['y_shape']
        cb_h, cb_w = metadata['cb_shape']
        cr_h, cr_w = metadata['cr_shape']

        # Восстановление каналов
        channels = []
        for ch, (channel_blocks, q_table, orig_shape) in enumerate(zip(
                blocks_data,
                [q_luma, q_chroma, q_chroma],
                [(y_h, y_w), (cb_h, cb_w), (cr_h, cr_w)]
        )):
            pad_h = (orig_shape[0] + 7) // 8 * 8
            pad_w = (orig_shape[1] + 7) // 8 * 8

            channel = np.zeros((pad_h, pad_w))
            idx = 0
            dc_prev = 0

            for i in range(0, pad_h, 8):
                for j in range(0, pad_w, 8):
                    if idx >= len(channel_blocks):
                        break

                    block_data = channel_blocks[idx]
                    dc_diff = block_data['dc_diff']
                    ac_rle = block_data['ac_rle']

                    # Восстановление DC
                    dc = dc_prev + dc_diff
                    dc_prev = dc

                    # Восстановление AC
                    quant_block = np.zeros((8, 8))
                    quant_block[0, 0] = dc  # DC компонент
                    ac_zigzag = rle_decode(ac_rle)
                    for k, coord in enumerate(ZIGZAG_ORDER[1:]):  # Пропускаем DC
                        if k < len(ac_zigzag):
                            quant_block[coord] = ac_zigzag[k]

                    # Обратное квантование и IDCT
                    dct_block = dequantize(quant_block, q_table)
                    block = inverse_dct(dct_block)

                    channel[i:i + 8, j:j + 8] = block
                    idx += 1

            # Обрезаем до оригинального размера
            channel = channel[:orig_shape[0], :orig_shape[1]]
            channels.append(channel)

        # Апсемплинг chroma каналов и конвертация в RGB
        y_channel = channels[0]
        cb_upsampled = upsample(channels[1], 2)[:y_h, :y_w]
        cr_upsampled = upsample(channels[2], 2)[:y_h, :y_w]

        ycbcr = np.stack((y_channel, cb_upsampled, cr_upsampled), axis=-1)
        ycbcr = np.clip(ycbcr, 0, 255).astype(np.uint8)
        image = Image.fromarray(ycbcr, 'YCbCr').convert('RGB')
        image.save(output_path)
        return image


def plot_compression_size_vs_quality(image_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    qualities = list(range(0, 101, 5))
    file_sizes = []

    for quality in tqdm(qualities):
        compressed_path = os.path.join(output_dir, f"compressed_{quality:03d}.bin")

        compressor = JPEGCompressor(quality=quality, mode='plot')
        compressor.compress(image_path, compressed_path)
        file_sizes.append(os.path.getsize(compressed_path))

    plt.figure(figsize=(12, 6))
    plt.plot(qualities, file_sizes, 'b-o', linewidth=1.5, markersize=5)

    # Настройки отображения
    plt.title(f"Зависимость размера сжатого файла от уровня качества\n{os.path.basename(image_path)}")
    plt.xlabel("Уровень качества сжатия")
    plt.ylabel("Размер файла (байты)")
    plt.grid(True, alpha=0.3)
    plt.xticks(range(0, 101, 10))
    plt.show()


if __name__ == "__main__":

    input_path = r"C:\py projects\pythonProject\Converted_images\Lenna.png"
    path_to_converted_images = r'C:\py projects\pythonProject\Converted_images'
    compressed_path = r"C:\py projects\pythonProject\compressed.bin" # <- путь для сохранения сжатого файла
    output_path = r"C:\py projects\pythonProject\decompressed.png"  # <- путь для сохранения разжатого файла
    output_dir = "Images_for_plot"

    image_name = input_path.split('\\')[-1].split('.')[0]
    suffixes = ['.png', '_grayscale.png', '_bw_dither.png', '_bw_threshold.png']
    """--------------------------------------------------------------------"""
    #Графики для основного изображения и сконвертированных изображений
    for suffix in suffixes:
        image_path = path_to_converted_images + '/' + image_name + suffix
        plot_compression_size_vs_quality(image_path, output_dir)

    """---------------------------------------------------------------------"""
    for quality in [0, 20, 40, 60, 80, 100]:
        compressor = JPEGCompressor(quality=quality)
        compressor.compress(input_path, compressed_path)

        decompressor = JPEGDecompressor()
        image = decompressor.decompress(compressed_path, output_path)
        plt.imshow(image)
        plt.show()
