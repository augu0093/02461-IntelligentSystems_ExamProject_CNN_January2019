# Optimizing preprocessing of images
---
## Base-program code *AUGMENTATION*
---
### Imports

    ```python
    #For preprocessing images
    from keras.preprocessing.image import ImageDataGenerator
    ```

---

### Augmentation of images

    ```python
        train_datagen = ImageDataGenerator(rescale=1./255,
                                          shear_range=0.2,
                                          zoom_range=0.2,
                                          horizontal_flip=True)

       test_datagen = ImageDataGenerator(rescale=1./255)
    ```

## Articles about image augmentation and data preprocessing
---
### [How to use Deep Learning when you have Limited Data](https://medium.com/nanonets/how-to-use-deep-learning-when-you-have-limited-data-part-2-data-augmentation-c26971dc8ced)
From https://medium.com/nanonets/how-to-use-deep-learning-when-you-have-limited-data-part-2-data-augmentation-c26971dc8ced

Almost linear relationship between complexity of model and the amount of data required. 

Image classification problems require large amounts of data

Transfer learning can help by providing a large dataset for training and is also pretrained. 

---

### [Data Augmentation | How to use Deep Learning when you have Limited Data — Part 2](https://medium.com/nanonets/how-to-use-deep-learning-when-you-have-limited-data-part-2-data-augmentation-c26971dc8ced)
From https://medium.com/nanonets/how-to-use-deep-learning-when-you-have-limited-data-part-2-data-augmentation-c26971dc8ced

(For coding blocks check article)

Getting the most out of a limited dataset

Data augmentation helps to increase the amount of relevant data in a dataset

Augmentation: flipping, zooming, shearing, rotating to synthetically create more images and therefore a larger dataset

#### Augmentations techniques
Offline augmentation: performing transformations beforehand therefore increasin the size of dataset. Prefered on smaller datasets.
Online augmentation: is prefered on larger datasets when you cannot afford explosive increase in size. Performs transformation on mini-batches to feed the model. Can be accelerated on the GPU. 

Flips: flipping the image both horisontally and vertically (DAF = 2 to 4x).

Rotation: rotating the image. The dimensions of the image may not be preserved after rotating (DAF = 2 to 4x).

Scaling: scaling the image outward og inward. Scaling inward reduces image size, is this zoom (DAF = Arbitrary).

Crop: Randonly sample a section form the original image and then resize the section to the original image size. Random cropping (DAF = Arbitrary).

Translation: moving the image along the X og Y direction (or both). Mostly relevant if the image has a background of solid color. Useful as most objects can be located almost anywhere in the image. "Forces" convolutional network to look everywhere (DAF = Arbitrary). 

Gaussian Noise: Gaussian Noise, which has zero mean, essentially has data points in all frequencies therefore distorting the high frequency features. Adding the right amount of noise can enhance the learning capability (DAF = 2x).

Conditional GANs: Generative Adversarial Networks. 

Interpolation: Assumptions about how to fill the "blank spots" when the dimensions of the augmented image does not match the original or intented image dimensions. These fills can be constant (fill w. constant value), edge (fill w. edge values), reflect (image pixel values reflected along image boundary), symmetric (fill w. copy of edge pixels), wrap (image is repeated beyond its boundary). 

#### Preprocessing the images correctly
Consider which augmentations that makes sense. Make sure  no to increase irrelevant data!

---

### [Data Augmentation on Images](https://towardsdatascience.com/data-augmentation-and-images-7aca9bd0dbe8)
From https://towardsdatascience.com/data-augmentation-and-images-7aca9bd0dbe8

Data augmentation: methods that enhance the data the we already have (flips, translations, color distortions, adding random noise).

Is done to utilize the full power of the convolutional neural network. 

Image transformation will result in a more robust model that learns better characteristics for making better distinctions between images. 

Helps learning with small amounts of data.

Helps prevent overfitting by creating differences in the dataset. 

---

### [Advanced Data Augmentation Strategies](https://towardsdatascience.com/advanced-data-augmentation-strategies-383226cd11ba)
From https://towardsdatascience.com/advanced-data-augmentation-strategies-383226cd11ba

No better way to quickly boost model performance than adding more training data to it. 

Using data augmentation to synthetically increase volume of data and enlarge the dataset. 

More advanced forms: stochastic region sharpening, elastic transforms, randomly erasing patches of the image and more.

Methods being researched: Adversarial Training, Generative Adversarial Networks, Style Transfer, Reinforcement Learning to search through the space of augmentation possibilities. 

Adversarial Training: Two models; one tries to classify and the other tries to fool the other classifier by adding noise to images. 

Generative Adversarial Networks: using a generator network to map a vector of noise into a channel image tensor to generate images that look like the original training images (problem: hard to produce high resolution output images). 

Neural Style Transfer: Extremely good at seperating style from content. The style representation is formatted as a "gram matrix" and a loss function is derived from this to transfer the style into another image while preserving the content. 

Alot of different parameters to adjust to find the optimal combination of data augmentation (SEE MODEL IN ARTICLE, "Tree", "Discrete Space Search").

---

### **PAPER** [The Effectiveness of Data Augmentation in Image Classification using Deep Learning](https://arxiv.org/abs/1712.04621)
From https://arxiv.org/abs/1712.04621

s1: " It is common knowledge that the more data an ML algorithm has access to, the more effective it can be."

s1: Meget data er godt, endda selvom en del af dataen er fejlagtig.

s1: "This approach has proven effective in multiple problems. Data augmentation guided by expert knowledge [14], more generic image augmentation [18], and has shown effective in image classification [16]."

s2: "The problem with small datasets is that models trained with them do not generalize well data from the validation and test set. Hence, these models suffer from the problem of overfitting."

s2: "Data augmentation is another way we can reduce overfitting on models, where we increase the amount of training data using information only in our training data."

s3: "Traditional transformations consist of using a combination of affine transformations to manipulate the training data [9]. For each input image, we generate a ”duplicate” image that is shifted, zoomed in/out, rotated, flipped, distorted, or shaded with a hue. Both image and duplicate are fed into the neural net. For a dataset of size N, we generate a dataset of 2N size.

===
Generally traditional augmentation (as described above) performs well in tests and is even capable of outperforming newer and more advanced forms of image augmentation (dogs vs cats study, 5.5%point better performance than GANs and 0.5-3.5%point than other neural networks.
===

## [ImageDataGenerator function (keras)](https://keras.io/preprocessing/image/)
From https://keras.io/preprocessing/image/

"Generate batches of tensor image data with real-time data augmentation. The data will be looped over (in batches)."

Traditional image augmentation. 

### ImageDataGenerator parameters
The following parameters are considered for the experiment:

#### No image augmentation
Running the program with limited data preprocessing and therefore no augmentation of images to show effectiveness of image augmentation (https://arxiv.org/abs/1712.04621). By doing this the size of the data set is hereby also smaller which should lead to a lower accuracy and a larger risk of overfitting. 

name: noAug

#### Only rotation, shearing, zooming etc. No color differences.
Only rotating, shearing, zooming, horizontal flipping, rescaling. No difference in color or brightness. 
Height and width shift.
Interpolation is set to "nearest" (`fill_mode='nearest'´), as the hand in most pictures is in the middle and with this setting the edges of the images repeats outside of the image and hopefully does not put more hands or fingers in the image. 

name: basicAug

#### Implementing channel shifting, white noise and adjusting brightness
Focusing on editing the images with white noise and channel shifting (RGB). No rotations etc. Brightness adjustments. 

name: colorAug

#### Standardization, normalization 1
Setting each sample mean to 0. Dividing each input by its std.

name: stdNormAug1

#### Standardization, normalization 2
Setting each input mean to 0 over the dataset feature-wise. Dividing each input by std of the dataset feature-wise. 

name: stdNormAug2

#### Standardization, normalization 3
Doing both of the above (1 and 2). 

name: stdNormAug3

#### Combination of best results
Combining the setting which had the best results during testing to hopefully perform a final optimization of the data preprocessing.

name: ...

---

### Setting thoughts
Using  ´´´width_shift_range=0.25´´´ and ´´´height_shift_range=0.1´´´ because of feedback from the live predicter through the camera. Problem with classifying in all of the webcam area. The two function translates in the X and Y directions. 

