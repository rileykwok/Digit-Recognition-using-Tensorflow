# Digit-Recognition-using-Tensorflow
Using a deep learning model to predict hand written digits

# About
The data files are from [MNIST](http://yann.lecun.com/exdb/mnist/). It contains gray-scale images of hand-drawn digits, from zero through nine.

Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total. Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel, with higher numbers meaning darker. This pixel-value is an integer between 0 and 255, inclusive.


# Exploratory Data Analysis
The labels distribution of the images. 

![image](https://github.com/rileykwok/Digit-Recognition-using-Tensorflow/blob/master/images/digit%20label%20distribution.png)

The labels are quite evenly distributed.
Have a look at some of the digits:

![image](https://github.com/rileykwok/Digit-Recognition-using-Tensorflow/blob/master/images/digit%20example.png)


# Data Augmentation
In order to avoid overfitting problem, we need to expand artificially our handwritten digit dataset. We can make your existing dataset even larger. The idea is to alter the training data with small transformations to reproduce the variations occuring when someone is writing a digit.

By applying just a couple of these transformations to our training data, we can easily double or triple the number of training examples and create a very robust model.

```python
datagen = ImageDataGenerator(
        rotation_range=10,       # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1,        # Randomly zoom image 
        width_shift_range=0.1,   # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1)  # randomly shift images vertically (fraction of total height)

datagen.fit(x_train)
```
The improvement is important :

Without data augmentation: accuracy of 98.1%
<br>With data augmentation: accuracy of 99.4%


# Learning rate optimizer and annealer
In order to make the optimizer converge faster and closest to the global minimum of the loss function, an annealing method of the learning rate (LR) is used.

The higher LR, the bigger are the steps and the quicker is the convergence. However the sampling is very poor with an high LR and the optimizer could probably fall into a local minima.

Its better to have a decreasing learning rate during the training to reach efficiently the global minimum of the loss function.

To keep the advantage of the fast computation time with a high LR, i decreased the LR dynamically every X steps (epochs) depending if it is necessary (when accuracy is not improved).

With the ReduceLROnPlateau function from Keras.callbacks, i choose to reduce the LR by half if the accuracy is not improved after 3 epochs.

```python
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
```

# Keras Model
The Keras Sequential model is used, as this is relatively easy to use and deploy, adding layers one by one.

The first is the convolutional (Conv2D) layer. It is like a set of learnable filters. Each filter transforms a part of the image (defined by the kernel size) using the kernel filter. The kernel filter matrix is applied on the whole image. Filters can be seen as a transformation of the image.

The second important layer in CNN is the pooling (MaxPool2D) layer. This layer simply acts as a downsampling filter. It looks at the 2 neighboring pixels and picks the maximal value. These are used to reduce computational cost, and to some extent also reduce overfitting. We have to choose the pooling size (i.e the area size pooled each time) more the pooling dimension is high, more the downsampling is important.

Combining convolutional and pooling layers, CNN are able to combine local features and learn more global features of the image.

Dropout is a regularization method, where a proportion of nodes in the layer are randomly ignored (setting their wieghts to zero) for each training sample. This drops randomly a propotion of the network and forces the network to learn features in a distributed way. This technique also improves generalization and reduces the overfitting.

'relu' is the rectifier (activation function max(0,x). The rectifier activation function is used to add non linearity to the network.

The Flatten layer is use to convert the final feature maps into a one single 1D vector. This flattening step is needed so that you can make use of fully connected layers after some convolutional/maxpool layers. It combines all the found local features of the previous convolutional layers.

The dense layer is just a regular layer of neurons in a neural network. Each neuron recieves input from all the neurons in the previous layer, thus densely connected. In the last layer(Dense(10,activation="softmax")) outputs distribution of probability of each class.

Lastly compile all the layers to form a Keras neural network model.

``` Python
model = Sequential()
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.25))
model.add(Dense(10, activation = "softmax"))

model.compile(optimizer = "RMSprop" , loss = "categorical_crossentropy", metrics=["accuracy"])

```
The accuracy and loss of the train and test set is shown below:

![image](https://github.com/rileykwok/Digit-Recognition-using-Tensorflow/blob/master/images/model%20history.png)


# Evaluation
Let's evaluate the model.
Confusion matrix:

![image](https://github.com/rileykwok/Digit-Recognition-using-Tensorflow/blob/master/images/evaluation.png)

Here we can see that our CNN performs very well on all digits with few errors considering the size of the validation set.

However, it seems that our CNN has some little troubles with the 4 digits, hey are misclassified as 9. Sometime it is very difficult to catch the difference between 4 and 9 when curves are smooth.

Let's investigate the most important errors . For that purpose we plot some of the mis-classified digits where the difference between the probabilities of real value and the predicted ones in the results are the largest.

![image](https://github.com/rileykwok/Digit-Recognition-using-Tensorflow/blob/master/images/wrong%20predictions.png)

The most important errors are also the most intrigous.

The model is not ridiculous. Some of these errors can also be made by humans, especially for one the 9 that is very close to a 4. 


# Results
The final Keras model is able to achieve a 99.4% accuracy.

