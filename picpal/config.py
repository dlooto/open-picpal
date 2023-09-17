import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


TRAIN_DATA_DIR = os.path.join(PROJECT_ROOT, 'data/train')
VALIDATION_DATA_DIR = os.path.join(PROJECT_ROOT, 'data/validation')
SAVED_MODEL_DIR = os.path.join(PROJECT_ROOT, 'data/saved_model')
MODEL_FILE_NAME = 'saved_model.h5'


# Hyper-parameters: dimensions of InceptionV3.
IMG_WIDTH, IMG_HEIGHT = 256, 256
BATCH_SIZE = 32
EPOCHS = 20


# the number of class, you can add more classification labels here.
CLASS_LABELS = {
    0: "cat",
    1: "movie",
    2: "book",
    3: "digit",
}


def get_train_img_path(img_name):
    return os.path.join(TRAIN_DATA_DIR, img_name)


def get_validation_img_path(img_name):
    return os.path.join(VALIDATION_DATA_DIR, img_name)


def get_model_path():
    return os.path.join(SAVED_MODEL_DIR, MODEL_FILE_NAME)
