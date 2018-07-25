#coding=utf-8
import random
from os import path
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from captcha.image import ImageCaptcha

from os.path import join

number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
            'V', 'W', 'X', 'Y', 'Z']
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']
charset=number + alphabet + ALPHABET

def random_captcha_text(
        char_set=charset,
        captcha_size=4):
    """
    :param char_set:
    :param captcha_size:
    :return:
    """
    captcha_text = []
    for i in range(captcha_size):
        c = random.choice(char_set)
        captcha_text.append(c)
    return captcha_text


def gen_captcha_text_and_image():
    """
    生成字符对应的验证码
    :return:
    """
    image = ImageCaptcha()

    captcha_text = random_captcha_text(captcha_size=random.choice([4,5,6,7,8]))
    captcha_text = ''.join(captcha_text)

    captcha = image.generate(captcha_text)

    captcha_image = Image.open(captcha)
    captcha_image = np.array(captcha_image)
    return captcha_text, captcha_image


def wrap_gen_captcha_text_and_image():
    """
    有时生成图像大小不是(60, 160, 3)
    :return:
    """
    while True:
        text, image = gen_captcha_text_and_image()
        if image.shape != (60, 160, 3):
            continue
        return text, image


def __gen_and_save_image(dir_name,data_len):
    """
    可以批量生成验证图片集，并保存到本地，方便做本地的实验
    :return:
    """

    for i in range(data_len):
        text, image = wrap_gen_captcha_text_and_image()

        im = Image.fromarray(image)
        image_name = '%s_%s.png' % (text, str(i))

        img_root = join('img', dir_name)
        image_file = path.join(img_root, image_name)
        im.save(image_file)


def __demo_show_img():
    """
    使用matplotlib来显示生成的图片
    :return:
    """
    text, image = wrap_gen_captcha_text_and_image()

    print("验证码图像channel:", image.shape)  # (60, 160, 3)

    f = plt.figure()
    ax = f.add_subplot(111)
    ax.text(0.1, 0.9, text, ha='center', va='center', transform=ax.transAxes)
    plt.imshow(image)

    plt.show()


if __name__ == '__main__':
    __gen_and_save_image('train',1000)
    __gen_and_save_image('val',200)
    __gen_and_save_image('infer',50)
