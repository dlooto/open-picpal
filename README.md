# OpenPicPal

<br>

[中文文档](README_CN.md) | [English README](README.md)

<br>

OpenPicPal is an open-source tool for image training and automatic classification.

As a Python-based project, the goal of OpenPicPal is to train and acquire a necessary classification model utilizing the InceptionV3 base-model and a previously prepared image dataset, followed by automating image classification using the resultant model.

This image classification project involves the following key steps:
1. Choose a Base Model: For this project, InceptionV3 serves as the base model (considering its balance between accuracy and runtime performance, with not too many parameters).
2. Prepare Image Data: This includes the preparation of image datasets for training, validation, testing, etc.
3. Train the Model: Generate the required result model.
4. Image Classification/Prediction: Utilize the resulting model for image classification/prediction.

## Data Preparation
The source of image data can be obtained by searching for open-source datasets on the internet or by writing custom web crawlers.
Prepare the image data and organize it into the following directory structure:
```
data/
    |__saved_model/     # Directory for saving trained model files
    |__train/           # Directory for training image datasets categorized into subdirectories
        |__book
        |__cat
        |__digit
        |__movie
        ……
    |__validation/      # Validation image datasets, ensuring the same subdirectory structure as 'train'
```

#### Image Dataset Configuration
* Each subdirectory under 'train' should contain image files corresponding to the respective categories (e.g., 'book/' directory contains all images related to books).
* Prepare as many subdirectories as there are categories.
* Select approximately 1/4 to 1/3 of the images from 'train' subdirectories and place them in corresponding 'validation' subdirectories. For example, if there are 100 images in 'train/book/', consider moving 25 images to 'validation/book/' to create the validation dataset.
* Ensure that the number of subdirectories in 'validation' matches the number in 'train', and maintain a certain proportion (e.g., validation dataset size is 1/4 to 1/3 of the training dataset).
* Generally, more images lead to better classification performance, but consider training time and performance when deciding on the dataset size.
* If you have new categories, add new subdirectories in 'train' and 'validation'.

## Development Environment
0. Required python libraries:
```shell
    python==3.9.2
    keras==2.11.0
    tensorflow==2.11.0
```

1. Clone code: 
```
git clone git@github.com:dlooto/open-picpal.git
```
2. Navigate to the project root directory: `cd open-picpal`
3. Create the data directory structure as described in "Data Preparation." The `data/saved_model/` directory is used to store trained model file.
4. Create and modify configuration files:
```shell
cp open-picpal/config.py.example  open-picpal/config.py     # Copy the example file
vi open-picpal/config.py         # Modify the relevant parameters
```
5. Install Python libraries using pip:
```shell
pip install -r requirements.txt
```

## Training and Classification
1. Set Class Labels:
```python
# picpal/config.py
CLASS_LABELS = {
    0: "book",
    1: "cat",
    2: "digit",
    3: "movie",
}
```

* Ensure that the `CLASS_LABELS` configuration in config.py matches the subdirectory names in 'train' and 'validation.' You can modify the label "book" to "books," for example, but make sure to update the subdirectory names in both 'train' and 'validation' accordingly.

2. Modify Training Parameters:
```python
MODEL_FILE_NAME = 'new_model.h5'    # Model file name used for image classification

IMG_WIDTH, IMG_HEIGHT = 256, 256     
BATCH_SIZE = 32                      
EPOCHS = 20                         
```
* The `(IMG_WIDTH, IMG_HEIGHT)` parameters set the required input image size for the InceptionV3 model. Adjust these dimensions based on your specific business requirements. For example, if your business primarily deals with portrait images (where the height is much larger than the width), set `IMG_HEIGHT` to be greater than `IMG_WIDTH` (or a multiple of `IMG_WIDTH`).
* The `EPOCHS` represents the number of iterations over the dataset during training. For instance, if you set `EPOCHS` to 10, training will iterate over the entire dataset 10 times.
* During each epoch, the dataset is typically divided into batches for processing. The `BATCH_SIZE` determines the number of samples in each batch.
* Different settings for `EPOCHS` and `BATCH_SIZE` will affect training duration.

3. Train the Model:
```shell
python -m picpal.train
```

4. Image Classification:
```shell
python -m picpal.predict
```


## Using as a Library
You can also use OpenPicPal as a Python library in other business code.

1. Build and publish:
```
python setup.py sdist bdist_wheel
```

2. Find the generated package (e.g., open-picpal-0.1.0.tar.gz) in the 'dist' directory and install it:
```
pip install dist/open-picpal-0.1.0.tar.gz
```

3. Using in business code:
```python
from picpal.train import Trainer

# Train the model
trainer = Trainer(
    epochs=20,
    batch_size=32, 
    img_width=256, 
    img_height=256
)
trainer.train()


# Image Classification
from picpal.config import *
from picpal.predict import Predictor

predictor = Predictor(
    model_path=get_model_path(),
    class_labels=CLASS_LABELS,
    img_width=IMG_WIDTH,
    img_height=IMG_HEIGHT
)

img_path_list = [
    "cat/14694fdd.png",
    "movie/e08b23f.png",
    "book/33h07bu31.jpg",
]

for i in img_path_list:
    result = predictor.predict_with_image_path(
        get_validation_img_path(i)
    )
    print(result, i)
```
