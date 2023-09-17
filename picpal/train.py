import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from .config import *


class Trainer:
    """
    A model trainer that encapsulates the building of the model,
    data augmentation, and model training.
    """
    def __init__(
        self,
        class_labels=None,
        train_data_dir=None,
        validation_data_dir=None,
        saved_model_dir=None,
        batch_size=None,
        epochs=None,
        img_width=None,
        img_height=None,
        model_file_name=MODEL_FILE_NAME
    ):
        """
        :param class_labels: list[str], optional. The list of class labels for the images.
        :param train_data_dir: str, optional. The directory for the training data.
        :param validation_data_dir: str, optional.The directory for the validation data.
        :param saved_model_dir: str, optional. The directory to save the trained model.
        :param batch_size: int, optional. The number of samples per batch during training.
        :param epochs: int, optional. The number of epochs for training.
        :param img_width: int, optional. The width of the images during training and validation.
        :param img_height: int, optional. The height of the images during training and validation.
        :param model_file_name: str, optional. The filename to save the trained model.
        """

        self.class_labels = class_labels if class_labels else CLASS_LABELS
        self.train_data_dir = train_data_dir if train_data_dir else TRAIN_DATA_DIR
        self.validation_data_dir = validation_data_dir if validation_data_dir else VALIDATION_DATA_DIR
        self.saved_model_dir = saved_model_dir if saved_model_dir else SAVED_MODEL_DIR
        self.model_file_name = model_file_name if model_file_name else MODEL_FILE_NAME

        self.batch_size = batch_size if batch_size else BATCH_SIZE
        self.epochs = epochs if epochs else EPOCHS
        self.img_width = img_width if img_width else IMG_WIDTH
        self.img_height = img_height if img_height else IMG_HEIGHT

        self.model = self.build_model()
        self.train_generator = self.build_train_generator()
        self.validation_generator = self.build_validation_generator()

        print("The number of training images:", self.train_generator.samples)
        print("The number of validation images:", self.validation_generator.samples)

    def build_validation_generator(self):
        valid_datagen = ImageDataGenerator(rescale=1. / 255)
        validation_generator = valid_datagen.flow_from_directory(
            self.validation_data_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical'
        )
        print("validation classes: ", validation_generator.class_indices)
        return validation_generator

    def build_train_generator(self):
        # Create an ImageDataGenerator for training data.
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,  # rescale image pixel values to 0-1 for prediction
            shear_range=0.2,  # degree of shear transformation
            zoom_range=0.2,  # range of random zoom
            horizontal_flip=True  # randomly flip images horizontally
        )

        # The generator will read images found in the "TRAIN_DATA_DIR" directory and
        # generate batches of augmented image data. It uses the `flow_from_directory()`
        # method to directly generate data and labels from the images.
        train_generator = train_datagen.flow_from_directory(
            self.train_data_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical'
        )
        print("train classes: ", train_generator.class_indices)
        return train_generator

    def build_model(self):
        """ Build a completed model """

        base_model = InceptionV3(weights='imagenet', include_top=False)

        # add global pooling layer and a fully connected layer
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)

        # add a classifier, assuming we have num_classes classes
        predictions = Dense(len(self.class_labels), activation='softmax')(x)

        # build the full model
        model = Model(inputs=base_model.input, outputs=predictions)

        # freeze all InceptionV3 layers
        for layer in base_model.layers:
            layer.trainable = False

        # compile the model(Operate after lock layer)
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=["accuracy"])
        return model

    def train(self):
        print("\nStart training model...")
        self.model.fit(
            self.train_generator,
            steps_per_epoch=self.train_generator.samples // self.batch_size,
            epochs=self.epochs,
            validation_data=self.validation_generator,
            validation_steps=self.validation_generator.samples // self.batch_size
        )
        self.model.save(
            os.path.join(self.saved_model_dir, self.model_file_name)
        )
        print(f"Model saved. ({self.model_file_name})")


def test_model_train():
    Trainer().train()


if __name__ == '__main__':
    test_model_train()
