import argparse
from io import StringIO
from PIL import Image, ImageEnhance
import os
import math

class M8AImage:
    image = Image.Image
    gamma = False
    palette = [
        [0, 0, 0],
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0],
        [255, 0, 255],
        [0, 255, 255],
        [255, 255, 255],
    ]
    counter = 0
    saturation = 2.0

    def loadImage(self, filepath):
        self.image = Image.open(filepath).convert('RGB')

    def resize_image_with_aspect_ratio(self, image, target_size, fill_color=(0, 0, 0)):
        width, height = image.size
        target_width, target_height = target_size

        # アスペクト比を維持したまま、指定のサイズ内に収まるようリサイズ
        aspect_ratio = min(target_width / width, target_height / height)
        new_width = int(width * aspect_ratio)
        new_height = int(height * aspect_ratio)
        resized_image = image.resize((new_width, new_height), Image.ANTIALIAS)

        # 画像範囲外の部分を黒(指定色)にしておく
        padded_image = Image.new('RGB', target_size, fill_color)
        x_offset = (target_width - new_width) // 2
        y_offset = (target_height - new_height) // 2
        padded_image.paste(resized_image, (x_offset, y_offset))

        return padded_image


    def resize(self, width, height):
        self.image = self.resize_image_with_aspect_ratio(self.image, (width, height))

    def fix_gamma(self, image):
        width, height = image.size

        for y in range(height):
            for x in range(width):
                r, g, b = image.getpixel((x, y))
                r = int(math.pow(float(r) / 255, 2.2) * 255)
                g = int(math.pow(float(g) / 255, 2.2) * 255)
                b = int(math.pow(float(b) / 255, 2.2) * 255)
                image.putpixel((x, y), (r, g, b))

    def reduce8(self, image):
        width, height = image.size
        output_image = image.copy()

        for y in range(height):
            for x in range(width):
                r, g, b = image.getpixel((x, y))
                r = self.genshoku8(r)
                g = self.genshoku8(g)
                b = self.genshoku8(b)
                output_image.putpixel((x, y), (r, g, b))
        return output_image

    def genshoku8(self, num):
        if num < 128:
            return 0
        else:
            return 255

    def od4x4(self, image):
        width, height = image.size
        if width % 4 != 0:
            return
        elif height % 4 != 0:
            return

        output_image = image.copy()

        for y in range(0, height, 4):  # クライアント領域Yサイズ(縦)
            for x in range(0, width, 4):  # クライアント領域Xサイズ(横)
                self.ordered_dither(output_image, x, y, 0*16)
                self.ordered_dither(output_image, x+1, y, 8*16)
                self.ordered_dither(output_image, x+2, y, 2*16)
                self.ordered_dither(output_image, x+3, y, 10*16)
                self.ordered_dither(output_image, x, y+1, 12*16)
                self.ordered_dither(output_image, x+1, y+1, 4*16)
                self.ordered_dither(output_image, x+2, y+1, 14*16)
                self.ordered_dither(output_image, x+3, y+1, 6*16)
                self.ordered_dither(output_image, x, y+2, 3*16)
                self.ordered_dither(output_image, x+1, y+2, 11*16)
                self.ordered_dither(output_image, x+2, y+2, 1*16)
                self.ordered_dither(output_image, x+3, y+2, 9*16)
                self.ordered_dither(output_image, x, y+3, 15*16)
                self.ordered_dither(output_image, x+1, y+3, 7*16)
                self.ordered_dither(output_image, x+2, y+3, 13*16)
                self.ordered_dither(output_image, x+3, y+3, 5*16)

        return output_image

    def ordered_dither(self, image, x, y, a):
        r, g, b = image.getpixel((x, y))
        r = 0 if r < a else 255
        g = 0 if g < a else 255
        b = 0 if b < a else 255
        image.putpixel((x, y), (r, g, b))


    def error_diffusion_fs(self, image):
        width, height = image.size
        output_image = image.copy()

        for y in range(height):

            if y % 2 == 0:
                x_range = range(width)
            else:
                x_range = range(width - 1, -1, -1)

            for x in x_range:
                r, g, b = output_image.getpixel((x, y))
                if r > 127:
                    r_new, er = 255, r - 255
                else:
                    r_new, er = 0, r

                if g > 127:
                    g_new, eg = 255, g - 255
                else:
                    g_new, eg = 0, g

                if b > 127:
                    b_new, eb = 255, b - 255
                else:
                    b_new, eb = 0, b

                output_image.putpixel((x, y), (r_new, g_new, b_new))
                coords = [(x + 1, y), (x - 1, y + 1), (x, y + 1), (x + 1, y + 1)]
                errors = [(er * 7 / 16, eg * 7 / 16, eb * 7 / 16),
                          (er * 3 / 16, eg * 3 / 16, eb * 3 / 16),
                          (er * 5 / 16, eg * 5 / 16, eb * 5 / 16),
                          (er * 1 / 16, eg * 1 / 16, eb * 1 / 16)]

                for (dx, dy), (de_r, de_g, de_b) in zip(coords, errors):
                    if 0 <= dx < width and 0 <= dy < height:
                        r_adj, g_adj, b_adj = output_image.getpixel((dx, dy))
                        r_adj = min(max(int(r_adj + de_r), 0), 255)
                        g_adj = min(max(int(g_adj + de_g), 0), 255)
                        b_adj = min(max(int(b_adj + de_b), 0), 255)

                        output_image.putpixel((dx, dy), (r_adj, g_adj, b_adj))

        return output_image

    def naive_sigmoid(self, x, gain, mid):
        return 1 / (1 + math.exp(-gain * (x - mid)))

    def naive_inverse_sigmoid(self, x, gain, mid):
        min_val = self.naive_sigmoid(0.0, gain, mid)
        max_val = self.naive_sigmoid(1.0, gain, mid)
        s = self.naive_sigmoid(x, gain, mid)
        a = (max_val - min_val) * x + min_val
        return math.log(1.0 / a - 1.0)

    def scaled_inverse_sigmoid(self, x, gain, mid):
        if x < 0.00001 or x >= 0.99999:
            return x
        min_val = self.naive_inverse_sigmoid(0.0, gain, mid)
        max_val = self.naive_inverse_sigmoid(1.0, gain, mid)
        s = self.naive_inverse_sigmoid(x, gain, mid)
        return (s - min_val) / (max_val - min_val)

    def sigmoidal_contrast(self, image, alpha, mid_point):
        lut = [int(self.scaled_inverse_sigmoid(i/255.0, alpha, mid_point) * 255) for i in range(256)]
        lut = lut * 3
        return image.point(lut)

    def ordered_dithering(self, im, level):
        width, height = im.size
        oim = Image.new("RGB", im.size)
    
        tw, th, _ = self.get_dither_table(256, 0)
    
        ods = []
        for i in range(level):
            v0 = (256 * i) // level
            v1 = (256 * (i + 1)) // level
            vrange = v1 - v0
            _, _, od = self.get_dither_table(vrange, v0)
            ods.append([v0, v1, od])
    
        src_pixel = im.load()
        dst_pixel = oim.load()
    
        for y in range(height):
            dy = y % th
            for x in range(width):
                dx = x % tw
                v = src_pixel[x, y]
                col = [0,0,0]
                for v0, v1, od in ods:
                    for i in range(3):
                        if v0 <= v[i] and v[i] <= v1:
                            if v[i] < od[dy][dx]:
                                col[i] = v0
                            else:
                                col[i] = v1
                dst_pixel[x, y] = tuple(col)
    
        return oim

    def get_dither_table(self, vrange, ofs):
        d = 17
        tbl = [
            [1, 9, 3, 11],
            [13, 5, 15, 7],
            [4, 12, 2, 10],
            [16, 8, 14, 6]
        ]
        if d >= vrange:
            print("Error: dither table range >= vrange")
            return
        w = len(tbl[0])
        h = len(tbl)
        odtbl = [[0 for i in range(w)] for j in range(h)]

        for y in range(h):
            for x in range(w):
                odtbl[y][x] = tbl[y][x] * vrange / d + ofs
        return w, h, odtbl

    def convert_retro(self, image, palette):
        input_image = image.convert("RGB")
        contrast_image = self.sigmoidal_contrast(input_image, -8, 0.5)
        modulated_image = ImageEnhance.Color(contrast_image).enhance(self.saturation)
        dithered_image = self.ordered_dithering(modulated_image, 9)

        pal_img = Image.new('P', (16, 16))

        NUM_ENTRIES_IN_PILLOW_PALETTE = 256
        num_bands = len("RGB")
        palettedata = sum(palette, [])
        num_entries_in_palettedata = len(palettedata) // num_bands
        palettedata.extend(palettedata[:num_bands]
                           * (NUM_ENTRIES_IN_PILLOW_PALETTE
                              - num_entries_in_palettedata))
        pal_img.putpalette(palettedata)
        dithered_image = dithered_image.quantize(palette=pal_img, dither=Image.NONE).convert("RGB")

        return dithered_image

    def output_m8a(self, filename, image, out_png = False):
        width, height = image.size

        if width % 8 != 0:
            # print("横方向ドット数が8の倍数ではありません。処理を中止します。")
            return
        if height % 8 != 0:
            # print("縦方向ドット数が8の倍数ではありません。処理を中止しますか？。")
            return

        image = self.reduce8(image)
        self.counter = 0

        r = [-1, -1, 0, 0]
        g = [-1, -1, 0, 0]
        b = [-1, -1, 0, 0]
        outputflag = 0

        encorded = ""

        encorded = bytearray()
        ptr = 0
        encorded.extend(b'M8A\x00')
        encorded.append(width // 8)
        encorded.append(height)
        ptr += 6

        for y in range(height):
            for x in range(0, width, 2):
                r[2], g[2], b[2] = image.getpixel((x, y))
                r[3], g[3], b[3] = image.getpixel((x + 1, y))
                if r[0] == r[2] and g[0] == g[2] and b[0] == b[2] and r[1] == r[3] and g[1] == g[3] and b[1] == b[3]:
                    if self.counter == 256:
                        if outputflag > 0:
                            encorded.append(int((self.x1color8int(r[1], g[1], b[1]) * 8) + self.x1color8int(r[0], g[0], b[0])))
                            encorded.append(255)
                            ptr += 2
                        else:
                            outputflag = 1
                        self.counter = 1
                    else:
                        self.counter += 1
                else:
                    if outputflag > 0:
                        self.set2(r, g, b, encorded)
                    else:
                        outputflag = 1
                    r[0], g[0], b[0], r[1], g[1], b[1] = r[2], g[2], b[2], r[3], g[3], b[3]
                    self.counter = 1
        if self.counter != 0:
            self.set2(r, g, b, encorded)

        fn = filename
        with open(fn, 'wb') as f:
            f.write(encorded)
        if out_png:
            fn = filename + ".png"
            image.save(fn)

    def set2(self, r, g, b, encorded):

        if encorded is None:
            encorded = []
        result = ""
        ptr = len(encorded)
        if self.counter == 1:
            encorded.append(int(64 + (self.x1color8int(r[1], g[1], b[1]) * 8) + self.x1color8int(r[0], g[0], b[0])))
        elif self.counter == 2:
            encorded.append(int(128 + (self.x1color8int(r[1], g[1], b[1]) * 8) + self.x1color8int(r[0], g[0], b[0])))
        elif self.counter == 3:
            encorded.append(int(192 + (self.x1color8int(r[1], g[1], b[1]) * 8) + self.x1color8int(r[0], g[0], b[0])))
        else:
            encorded.append(int((self.x1color8int(r[1], g[1], b[1]) * 8) + self.x1color8int(r[0], g[0], b[0])))
            encorded.append(int(self.counter - 1))

        return encorded

    def x1color8int(self, r, g, b):
        return int(self.x1color_sub(g, 0)) * 4 + int(self.x1color_sub(r, 0)) * 2 + int(self.x1color_sub(b, 0))

    def x1color_sub(self, a, hi):
        if a < 64:
            return "0"
        elif a < 128:
            return "0" if hi else "1"
        elif a < 192:
            return "1" if hi else "0"
        else:
            return "1"

    def saveM8A(self, mode, gamma, out_png, path):
        width, height = self.image.size

        if width % 8 != 0:
            print("Error: width must be a multiple of 8")
            return True

        if height % 8 != 0:
            print("Error: height must be a multiple of 8")
            return True

        if gamma:
            self.fix_gamma(self.image)

        if mode == "dither":
            image = self.od4x4(self.image)
        elif mode == "edfs":
            image = self.error_diffusion_fs(self.image)
        elif mode == "retro":
            image = self.convert_retro(self.image, self.palette)
        else:
            image = self.reduce8(self.image)

        self.output_m8a(path, image, out_png)
        return False

class M8AConverter:
    paths = []
    image = M8AImage()
    force_write = False
    saturation = 2.0

    def __init__(self, paths):
        self.paths = paths

    def load(self, path):
        # from_pathが存在しない場合はエラーを表示して戻る
        if not os.path.exists(path):
            print("file not found. " + path)
            return

        self.image.loadImage(path)

    def resize(self, width, height):
        self.image.resize(width, height)

    def convert(self, mode, gamma, out_png):
        if self.paths == None or len(self.paths) == 0:
            return True
        
        from_path = self.paths[0]
        from_ext = os.path.splitext(from_path)[1].lower()
        to_path = ''
        if len(self.paths) >= 2:
            to_path = self.paths[1]
        else:
            to_path = os.path.splitext(from_path)[0] + '.m8a'

        if os.path.exists(to_path) and not self.force_write:
            print("file already exists. " + to_path)
            return

        # from_pathが存在しない場合はエラーを表示して戻る
        if not os.path.exists(from_path):
            print("file not found. " + from_path)
            return

        # 事前にloadしておく事
        # self.image.loadImage(from_path)
        self.image.saturation = self.saturation
        if not self.image.saveM8A(mode, gamma, out_png, to_path):
            print("convert..." + from_path + " to " + to_path )

def parse_size(input_str):
    width, height = map(int, input_str.split('x'))
    return width, height

def main():
    parser = argparse.ArgumentParser(description='m8acnv M8A image converter Version 0.1.0 Copyright 2023 H.O SOFT Inc. / hex125(293)')

    parser.add_argument('-f', '--force', help='set force write ', action='store_true')
    parser.add_argument('-m', '--mode', choices=['reduce','dither','edfs','retro'], help='convert mode (default = reduce)', default='reduce')
    parser.add_argument('-g', '--gamma', help='fixed gamma (default = disable)', action='store_true')
    parser.add_argument('-p', '--png', help='output png file (default = disable)', action='store_true')
    parser.add_argument('-s', '--resize', type=parse_size, required=False, default = None, help='resize image (ex. 320x240) (default = disable)')
    parser.add_argument('-S', '--saturation', type=float, required=False, default = None, help='saturation (effective only in retro mode / default = None)')
    parser.add_argument('path', help='file path(s)', nargs="*")

    args = parser.parse_args()

    paths = args.path

    if len(paths) == 0:
        parser.print_help()
        exit()

    m8aconv = M8AConverter(paths)
    m8aconv.force_write = args.force

    m8aconv.load(paths[0])

    if args.resize != None:
        x,y = args.resize
        m8aconv.resize(x, y)
    
    if args.saturation != None:
        m8aconv.saturation = args.saturation

    if m8aconv.convert(args.mode, args.gamma, args.png):
        parser.print_help()
        exit()

if __name__=='__main__':
    main()
