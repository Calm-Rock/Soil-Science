#Code Implementation of a Deep CNN which was used in prediciton of soil organic carbon
#For each sample, 3 images were taken by teh setup.
# The hyper parameters can be tweked to get the best result.
#Required Libraries
import numpy as np
import pandas as pd
import os
from skimage.transform import resize
from skimage.io import imread


y=np.asarray(pd.read_csv('/home/somchakr/different_features_values_of_all_samples.csv')['carbon_percent']) #Path to csv file containing carbon percent values
Y=[]
for i in range(len(y)):
  a=np.full((3,1), y[i])
  Y.append(a)
  
y=np.concatenate(Y)
images = []
for image_path in sorted(os.listdir('/home/somchakr/Final_numbered_images_CNN')): #sorting out the images by number
    path='/home/somchakr/Final_numbered_images_CNN'+ '/' + image_path
    print(path)
    img = imread(path)
    images.append(img)

X=np.array(images)    
print(X[0].shape)
from sklearn.model_selection import train_test_split
#Splitting the data set as 75% training and 25% validation
X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size=0.25)

#Required Libraries
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D, Dropout,BatchNormalization
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
regressor = Sequential()

# Step 1 - Convolution
regressor.add(Conv2D(32, (3, 3), input_shape = (318,2304, 2950, 3), activation = 'relu'))

# Step 2 - Pooling
regressor.add(MaxPooling2D(pool_size = (2, 2)))
regressor.add(Conv2D(32, (3, 3), activation = 'relu'))
regressor.add(MaxPooling2D(pool_size = (2, 2)))
regressor.add(Conv2D(32, (3, 3), activation = 'relu'))
regressor.add(MaxPooling2D(pool_size = (2, 2)))
regressor.add(Conv2D(32, (3, 3), activation = 'relu'))
regressor.add(MaxPooling2D(pool_size = (2, 2)))
regressor.add(Conv2D(32, (3, 3), activation = 'relu'))
regressor.add(MaxPooling2D(pool_size = (2, 2)))
regressor.add(BatchNormalization())
regressor.add(Dropout(0.3))
regressor.add(Conv2D(32, (3, 3), activation = 'relu'))
regressor.add(MaxPooling2D(pool_size = (2, 2)))
regressor.add(Conv2D(32, (3, 3), activation = 'relu'))
regressor.add(MaxPooling2D(pool_size = (2, 2)))
regressor.add(Conv2D(32, (3, 3), activation = 'relu'))
regressor.add(MaxPooling2D(pool_size = (2, 2)))
regressor.add(BatchNormalization())
regressor.add(Dropout(0.3))
regressor.add(Conv2D(32, (3, 3), activation = 'relu'))
regressor.add(MaxPooling2D(pool_size = (2, 2)))
regressor.add(Conv2D(32, (3, 3), activation = 'relu'))
regressor.add(MaxPooling2D(pool_size = (2, 2)))
regressor.add(Conv2D(32, (3, 3), activation = 'relu'))
regressor.add(MaxPooling2D(pool_size = (2, 2)))
regressor.add(BatchNormalization())
regressor.add(Dropout(0.3))

# Step 3 - Flattening
regressor.add(Flatten())

# Step 4 - Full connection
regressor.add(Dense(units = 8, activation = 'relu'))
regressor.add(Dense(units = 1, activation = 'linear'))

# Compiling the CNN
regressor.compile(optimizer = 'adam', loss = 'mse', metrics = ['mae'])
regressor.fit(X_train, y_train, batch_size =16, epochs =20)

# Predicting the carbon values
y_pred_test = regressor.predict(X_test)
Y_test=pd.DataFrame(data=y_test, columns=['Y_TEST'])
Y_pred_test=pd.DataFrame(data=y_pred_test, columns=['Y_PRED_TEST'])
Y=pd.concat([Y_test, Y_pred_test], axis =1)
# Saving the results 
Y.to_csv('/home/somchakr/results_full.csv') 
