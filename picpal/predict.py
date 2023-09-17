
import logging

import numpy as np

from keras.models import load_model
from keras import backend as K
from tensorflow.keras.preprocessing import image
from PIL import ImageFile
from PIL import Image
ImageFile.LOAD_TRUNCATED_IMAGES = True

from .config import *
from .utils import times

logger = logging.getLogger(__name__)


class Predictor:

    def __init__(
            self,
            model_path=None,
            class_labels=CLASS_LABELS,
            img_width=None,
            img_height=None,
    ):
        self.class_labels = class_labels
        self.img_width = img_width
        self.img_height = img_height

        print(f"(img_width, img_height) in predictor: {img_width, img_height}")
        if K.image_data_format() == 'channels_first':
            self.input_shape = (3, img_height, img_width)
        else:
            self.input_shape = (img_height, img_height, 3)

        # load the saved model
        print("Loading the existing model ...")
        start_time = times.now()
        mpath = model_path if model_path else get_model_path()
        self.model = load_model(mpath)
        print(f"Model loaded.(cost:{times.time_cost(start_time)}, {mpath})")

    def predict_with_image_path(self, img_path):
        """
        # Analyze images based on the input image path and make a classification.
        :param img_path: local full path of the image
        :return: image category
        """
        return self.predict_with_image(
            image.load_img(img_path, target_size=self.input_shape)
        )

    def predict_with_image(self, img):
        """
        # according to the input image, analyzing the image and make a classification
        :param img: a PIL.Image.Image object
        :return: image category, tuple(category_label, predicted_probability) type
        """
        assert isinstance(img, Image.Image)

        # resize the image
        img = img.resize((self.img_width, self.img_height))

        # image preprocessing, maintain consistency with training
        x = image.img_to_array(img) / 255.0
        x = np.expand_dims(x, axis=0)

        # perform classification
        preds = self.model.predict(x)
        logger.debug('Predicted probability:', preds)     # print predicted probability
        class_indices = np.argmax(preds, axis=1)
        logger.debug('predicted_class_indices:', class_indices)  # print class-label's `int` value

        return self.class_labels[class_indices[0]], preds[0][class_indices[0]]

    def predict_with_text(self, text):
        pass

    def predict_with_image_base64(self, image_base64):
        pass


def test_predict_with_image_path():
    img_path_list = [
        "cat/1e7a7b3582a5df48b64c677814694fdd.gif",
        "movie/7d12b30b652d4e08b23f.png",
        "book/2023-04-20_18-20-31.png",
        "digit/2023-04-20_19-37-47.png",
    ]

    print("Start predicting ...")
    predictor = Predictor(
        model_path=get_model_path(),
        class_labels=CLASS_LABELS,
        img_width=IMG_WIDTH,
        img_height=IMG_HEIGHT
    )

    for i in img_path_list:
        start_time = times.now()
        result = predictor.predict_with_image_path(
            get_validation_img_path(i)
        )
        print(result, f" Cost: {times.time_cost(start_time)}", i)


if __name__ == '__main__':
    test_predict_with_image_path()

