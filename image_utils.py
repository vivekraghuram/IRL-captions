import urllib
import tempfile
import os
from scipy.misc import imread
from preprocessing.image_feature_extraction import load_image
from keras.preprocessing import image
import numpy as np

import skimage
import skimage.transform
import skimage.io

import matplotlib.cm as cm
import matplotlib.pyplot as plt


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
        print ('URL Error: ', e.reason, url)
    except urllib.error.HTTPError as e:
        print ('HTTP Error: ', e.code, url)


def visualize_attention(im_path, alphas, words):
    """
    Reference here: https://github.com/kelvinxu/arctic-captions/blob/master/alpha_visualization.ipynb
    """
    pil_img = load_image(im_path)
    img = image.img_to_array(pil_img).astype(np.uint8)

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
            alpha_img = create_alpha_img(alpha, shape=alpha.reshape(7, 7), upscale=32)

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


def visualize_padded_relevancy(im_path, relevancy_map):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    rel_shape = 4
    assert relevancy_map.shape == (4, 4)

    expected_size = 224
    padded_size = 256

    rel = sigmoid(relevancy_map * -1)

    pil_img = load_image(im_path)
    img = image.img_to_array(pil_img).astype(np.uint8)

    padded_img = np.ones((padded_size, padded_size, 3), dtype=np.uint8)
    diff = padded_size - expected_size

    padded_img[:expected_size, :expected_size, :] = img
    padded_img[expected_size:padded_size, :expected_size, :] = img[-diff:, :, :]
    padded_img[:expected_size, expected_size:padded_size, :] = img[:, -diff:, :]
    padded_img[expected_size:padded_size, expected_size:padded_size] = img[-diff:, -diff:, :]

    alpha_img = create_alpha_img(rel, shape=(rel_shape, rel_shape), upscale=64, sigma=10)

    plt.subplot(1, 2, 1)
    plt.imshow(padded_img)
    plt.axis('off')

    plt.subplot(1, 2, 2)

    plt.imshow(padded_img)
    plt.imshow(alpha_img, alpha=0.8)
    plt.set_cmap(cm.Greys_r)
    plt.axis('off')
    plt.show()
