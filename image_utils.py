import urllib
import tempfile
import os
from scipy.misc import imread

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