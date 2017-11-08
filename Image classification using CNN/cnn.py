# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential #becasue CNN is still a sequence of layers, we need sequential package
from keras.layers import Conv2D #Since working on 2d images, we use this package
from keras.layers import MaxPooling2D #we used to proceed to step 2 to do pooling
from keras.layers import Flatten #used to convert all pooled featured maps into large feature vector( input of fully connected layer)
from keras.layers import Dense #used to add fully connected layers in a classic ANN
from keras.layers import Convolution2D

# Initialising the CNN
    #creating an object of sequential class to classify some images to tell eacg image is a feature of dog or cat
classifier = Sequential() 

# Step 1 - Convolution
    #adding the different layers
    #we are the 1 of 4 steps: i.e CONVOLUTION STEP, In this we are applying several feature detectors
    #We are checking the presence of specific feature in the input image and tabulating them resulting of FEATURE MAP
    #applying the method on this object, Conv2D(no_of_filters 32 for common practice, kernel_size,input_shape = (3, 256, 256) for color image as it have BGR 3 layers)
    #Rectifier function as RELU to increase non linearity
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu')) #(64,64,3 if tensorflow &&& 3,64,64 if using Theano as backend)

# Step 2 - Pooling
    #Pooling is reducing size of feature maps with taking Maximum as MaxPooling outputting as Pooled Feature Map
    #we call this pooled FMaps as pooling layer
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer (For the increase performance/accuracy)
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
    #Inputing the pool size for applying the max pooling on the feature maps
    #We are not losing the information by keeping it precise
    #We reducing the complexity by preserving the performance
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
    #We get huge single vector which is the input for ANN
    #Flatten() function flattens all the feature maps, we don't need to input any parameters as keras understand smartly
classifier.add(Flatten())

# Step 4 - Full connection
    #We manage to convert image into 1D, we need to use as a input vector as a input layer for classic ANN
    #We then need to create hidden layer or called as Fully Connected Layer
    #Units of Output_dim=128 roughly(no need to be exact), since it's a hidden layer we use RELU as activation function
classifier.add(Dense(units = 128, activation = 'relu'))
    #output_dim=1 and to return the probabilities of each task, we use softmax(if not binary)/Sigmoid(if binary)
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
    #We are compiling our CNN, optimizer='adam' algorithm for selecting stochastic gradient descent
    #loss='bin...opy' we use this for classification and secondly it's binary.. so we used this
    #if it's not a binary outcome we should use categorical_crossentropy
    #metrics='acc..'
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])




# Part 2 - Fitting the CNN to the images
    #Image augmentation pre-process the images to prevent the overfitting
    #https://keras.io/preprocessing/image/ for documentation on image pre processing
    #Image augmentation is the trick that selects the batches of images randomly and applying different operations on them to get the better results
    #ImageDataGenerator class allows to use the data generator functions.
from keras.preprocessing.image import ImageDataGenerator

    #Rescale is for image instead of scaling for data
    #object for ImageDataGenerator class, used to preprocess images
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2, #randomly a value
                                   horizontal_flip = True)
#object used to preprocess the images of test set
test_datagen = ImageDataGenerator(rescale = 1./255)
    
#Here we applied the image augmentation on the training set and resizing the images & by creating the batches
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64), #dimensions expected by CNN
                                                 batch_size = 32, #32 random samples of images go through the CNN after which the weights will be updated
                                                 class_mode = 'binary') #parameter indicating the dependent variable

#we are creating test set, resizing and providing batches
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32, 
                                           class_mode = 'binary')

#where we fit our CNN to the training set while also testing its performance on the test set
#
classifier.fit_generator(training_set,
                         steps_per_epoch = 8000, #no of images in our training set
                         epochs = 1, #No. of epochs used to train out CNN
                         validation_data = test_set, #Data on which we wanna evaluate our CNN
                         validation_steps = 2000) #no. of images in our test set



#INORDER TO STILL IMPROVE THE ACCURACY/PERFORMANCE OF THE MODEL
#we need to develop even deeper neural network and also by adding an other convolutional layer (the best solution)

# Part 3 - Making new predictions

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
    
    
import numpy as np
from keras.preprocessing import image
test_image1=image.load_img('dataset/single_prediction/cat_or_dog_2.jpg',target_size=(64,64))
test_image1=image.img_to_array(test_image1)
test_image1=np.expand_dims(test_image1,axis=0)
result1=classifier.predict(test_image1)
training_set.class_indices
if (result1[0][0]==1):
    predic='dog'
else:
    predic='cat'
