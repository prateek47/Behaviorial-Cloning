# Behaviorial Cloning Project

[//]: # (Image References)

[image1]: ./examples/distribution.png
[image2]: ./examples/center_2016_12_01_13_32_45_578.jpg
[image3]: ./examples/center_2016_12_01_13_33_15_644_rightturn.jpg
[image4]: ./examples/center_2016_12_01_13_31_13_584_leftturn.jpg


#### 1. Overview

The steps used to build this model are as following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road


This  repository includes the following files:
* clone.py containing the script to train the model
* drive.py for driving the car in autonomous mode
* model.h5 contains a trained convolution neural network

#### 2.  How to run the model ?
Using the Udacity simulator and drive.py function(provided by udacity), the car can be driven autonomously around the track by using the funcition(in the terminal)

```
python drive.py model.h5
```
#### 3. Project Code

The **clone.py** file contains the code for training and saving the convolution neural net model. The file contains comments to have better understanding of various steps undertaken during the training.

### Model Architecture and Training Strategy

#### 1. Model Inspiration

This model is based on the architecture of [Nvidia's end to end deep learning architecture](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) 
The model consist of :
1. 5 Convolution neural network layers with 3x3 and 5x5 filter sizes with min depth of 24 and max of 64.
2. Convolution layer are followed by 3 fully connected layers. 
3. Dropouts after first convolutional layer and first and second fully connected layer and second fully connected layer.
4. The data is Cropped using Keras cropping2d layer to remove the unused part in an image
5. The data is also Normalized using the Lambda function.

**Note:** Cropping & Normalization helps the image train faster

#### 2. Attempts to reduce overfitting in the model

The model consist of 3 dropout layers at various stages in the model, to reduce overfitting. Also, the model was tested on the simulator to enure the vehicle could stay on the track

#### 3. Model parameter tuning

'Adam' Optimizer is used to optimize learning rate automatically plus the batch size is kept at 128.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, A steering correction was added with images from left and right camera.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a simple flat Neural Network and check how the model was performing, then I slowly progressed to Adding a Convolutional layer and Normalizing the data. Seeing the Car moving towards left made me realize the left bias nature of the track, I flipped images and create the new augmented dataset, which not only help remove the left bias but also increase the size of the data.

Then I moved on to more sophisticated Nvidia end to end deep learning model.

I can see that the model had a low mean squared error on the training set but high on validation data, which showed that the model might be overfitting therefore, I addded dropout layers.

On the images, I can notice, that the top 75 pixel consisted of sky, mountain, extra terrain, which are useless in terms of teaching model how to make a turn also, the bottom 25 pixel just consisted of image dashboard, therefore, I removed the top 75 and bottom 25 pixels from the input images, this even helped in increasing the training time.

On running the model on the simulator, the vehicle was felling off th track at sharp turns and when lane disappered, therefore, to the strategy that best helped to tackle this challenge was to remove all images close to zero steering angle and having more evenly distributed dataset. Also, the right and left images were used with a correction factor on the steering angle.

#### 2. Final Model Architecture

The final model architecture consisted of:
1. Cropping the top 75 and bottom 2 pixels
2. Normalizing the images
3. 5 convolutional layers with kernal size 5x5 in first three and 3x3 in next two while depth varied from 24 to 64.
4. 3 fully connected layers
5. Dropout layers one in convolutional and other 2 in fully connected.

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, few laps were recorded on the track one using center lane driving. The key to success was data distribution. After the data was balanced by removing images closer to 0 degree angle the model started really well. The distribution looked liked this:

![alt text][image1]

Images that were being feeded to model with steering angle looked like below for different angles: Straight, Right turn, Straight and Left turn:


![alt text][image2] 

![alt text][image3] 

![alt text][image4]

To increase the size of the dataset, data was augmented. I flipped images and angles to balance the left biased turns in on the track.

I also used images from the other two cameras to teach model recovery with steering angles with correction factor 

The data was then finally shuffled with 20% kept for validation.

The validation set mainly helped in deteriming overfitting, while training set was used for training the model, various values for number of epochs was tested but in the end he ideal number of 10 was used as the mse loss is almost insignificant after that. 

Data usage was the key in the project!!!..

## Video

[![R (video)](https://youtu.be/22hXmoYl1YU)


