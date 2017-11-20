import numpy as np
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
from keras.preprocessing import image


def load_image(file_path, target_size=(224, 224)):
    return image.load_img(file_path, target_size=target_size)


def load_image_batch(file_paths, target_size=(224, 224)):
    return [load_image(f, target_size) for f in file_paths]


class BaseImageFeaturizer(object):

    def featurize(self, image):
        pass


class MockFeaturizer(BaseImageFeaturizer):
    def __init__(self, shape):
        self.shape = shape

    def featurize(self, img):

        return np.random.uniform(0, 1, tuple([len(img)]) + self.shape)


class VGG19Featurizer(BaseImageFeaturizer):

    def __init__(self, layer_name='fc2'):
        self.base_model = VGG19(weights='imagenet')
        self.model = Model(inputs=self.base_model.input, outputs=self.base_model.get_layer(layer_name).output)

    def featurize(self, img_input_batch):

        img_array_batch = np.array([image.img_to_array(img) for img in img_input_batch])
        pre_processed_batch = preprocess_input(img_array_batch)
        return self.model.predict(pre_processed_batch)

