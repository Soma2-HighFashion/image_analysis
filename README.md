## Street Snap Image Analysis

### 1.Gender Classifier

#### Classes
- 0 : Female
- 1 : Male

#### Model

1. **AlexNet** [paper](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)

	##### Event

	- Accuracy

	![images](images/gender_alexnet_accuracy.png)

	- Loss

	![images](images/gender_alexnet_loss.png)

	AlexNet is not enough!

2. **VGG**

	##### Event

	- Accuracy

	![images](images/gender_vgg_accuracy.png)

	- Loss

	![images](images/gender_vgg_loss.png)

### 2.Category Classifier

#### Classes
- 1 : Hiphop, Street
- 2 : Casual
- 3 : Classic, Suit
- 4 : Unique
- 5 : Sexy

1. **VGG**

	##### Event

	- Accuracy

	![images](images/category_vgg_accuracy.png)

	- Loss

	![images](images/category_vgg_loss.png)

2. **ResNet**

