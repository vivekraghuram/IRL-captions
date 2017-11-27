import urllib
import tempfile
import os
from scipy.misc import imread
from preprocessing.image_feature_extraction import load_image
from keras.preprocessing import image
import numpy as np
from scipy.special import expit
import skimage
import skimage.transform
import skimage.io

import matplotlib.cm as cm
import matplotlib.pyplot as plt

expected_size = 224
padded_size = 256

"""
    From image_utils.py from assignment3
"""


def image_from_url(url):
    """
    Read an image from a URL. Returns a numpy array with the pixel data.
    We write the image to a temporary file then read it back. Kinda gross.
    """
    try:
        f = urllib.request.urlopen(url)
        _, fname = tempfile.mkstemp()
        with open(fname, 'wb') as ff:
            ff.write(f.read())
        img = imread(fname)
        os.remove(fname)
        return img
    except urllib.error.URLError as e:
        print('URL Error: ', e.reason, url)
    except urllib.error.HTTPError as e:
        print('HTTP Error: ', e.code, url)


def visualize_attention(im_path, alphas, words):
    """
    Reference here: https://github.com/kelvinxu/arctic-captions/blob/master/alpha_visualization.ipynb
    """
    img = load_image_to_display(im_path)

    assert alphas.shape[0] > len(words)
    n_img = len(words) + 1
    r = 1
    c = 2

    for i in range(n_img):
        if i == 0:
            plt.subplot(r, c, 1)
            plt.imshow(img)
            plt.axis('off')
        else:
            pos = i % 2
            plt.subplot(r, c, pos + 1)

            lab = words[i - 1]
            plt.text(0, 1, lab, backgroundcolor='white', fontsize=13)
            plt.text(0, 1, lab, color='black', fontsize=13)

            alpha = alphas[i - 1]
            alpha_img = create_alpha_img(alpha, shape=(7, 7), upscale=32)

            plt.imshow(img)
            plt.imshow(alpha_img, alpha=0.8)
            plt.set_cmap(cm.Greys_r)
            plt.axis('off')

            if pos != 0 or i == n_img - 1:
                plt.show()
    plt.cla()


def create_alpha_img(alpha, shape, upscale, sigma=20):
    alpha_img = skimage.transform.pyramid_expand(alpha.reshape(shape), upscale, sigma=sigma)
    return alpha_img


def load_image_to_display(im_path):
    pil_img = load_image(im_path)
    return image.img_to_array(pil_img).astype(np.uint8)


def annotate_words(im_path, word_alpha_map, words):
    map_size = 4
    assert word_alpha_map.shape[0] == map_size
    assert word_alpha_map.shape[1] == map_size
    incr = padded_size / map_size

    sorted_arg = word_alpha_map.argsort()

    img = load_image_to_display(im_path)
    padded_img = pad_same_image(img, expected_size, padded_size)

    top_k = 2

    for k in range(top_k):
        plt.subplot(1, top_k, k + 1)
        plt.imshow(padded_img)
        plt.title("top@{}".format(k + 1))
        plt.axis("off")
        top_alpha_arg = sorted_arg[:, :, (k + 1) * -1]
        for i in range(map_size):
            for j in range(map_size):
                top_w = words[top_alpha_arg[i][j]]
                plt.text(incr * j, incr * i, top_w, rotation=-45, backgroundcolor='white', fontsize=10)
    plt.show()


def visualize_padded_relevancy(im_path, relevancy_map):

    need_padding_shape = 4

    img = load_image_to_display(im_path)

    if relevancy_map.shape == (need_padding_shape, need_padding_shape):
        image_to_show = pad_same_image(img, expected_size, padded_size)
        shape = (need_padding_shape, need_padding_shape)
        upscale = 64
    else:
        image_to_show = img
        shape = (7 , 7)
        upscale = 32

    rel = expit(relevancy_map * -1)
    alpha_img = create_alpha_img(rel, shape, upscale=upscale, sigma=10)

    plt.subplot(1, 2, 1)
    plt.imshow(image_to_show)
    plt.axis('off')

    plt.subplot(1, 2, 2)

    plt.imshow(image_to_show)
    plt.imshow(alpha_img, alpha=0.8)
    plt.set_cmap(cm.Greys_r)
    plt.axis('off')
    plt.show()


def pad_same_image(img, expected_size, padded_size):
    padded_img = np.ones((padded_size, padded_size, 3), dtype=np.uint8)
    diff = padded_size - expected_size
    padded_img[:expected_size, :expected_size, :] = img
    padded_img[expected_size:padded_size, :expected_size, :] = img[-diff:, :, :]
    padded_img[:expected_size, expected_size:padded_size, :] = img[:, -diff:, :]
    padded_img[expected_size:padded_size, expected_size:padded_size] = img[-diff:, -diff:, :]
    return padded_img
