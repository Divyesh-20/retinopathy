#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import the necessary packages

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import matplotlib.pyplot as plt
import PIL
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.utils import plot_model
from IPython.display import display
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, LearningRateScheduler


# In[2]:


from jupyterthemes import jtplot
jtplot.style(theme='monokai', context='notebook', ticks=True, grid=False) 


# In[5]:


os.listdir("C:\Users\shind\Downloads\codes\codes\dataset\archive\colored_images")


# In[6]:


os.listdir(os.path.join(r'C:\Users\shind\Downloads\codes\codes\dataset\archive\train.csv', r'C:\Users\shind\Downloads\codes\codes\dataset\archive\colored_images\Mild'))


# In[7]:


# Check the number of images in the dataset
train = []
label = []


# In[13]:


# os.listdir returns the list of files in the folder, in this case image class names
for i in os.listdir(r'C:\Users\shind\Downloads\codes\codes\dataset\archive\colored_images'):
  train_class = os.listdir(os.path.join(r'C:\Users\shind\Downloads\codes\codes\dataset\archive\colored_images', i))
  for j in train_class:
    img = os.path.join(r'C:\Users\shind\Downloads\codes\codes\dataset\archive\colored_images', i, j)
    train.append(img)
    label.append(i)

print('Number of train images : {} \n'.format(len(train)))


# In[14]:


train


# In[15]:


label




# In[18]:


# Visualize 5 images for each class in the dataset

fig, axs = plt.subplots(5, 5, figsize = (20, 20))
count = 0
for i in os.listdir(r'C:\Users\shind\Downloads\codes\codes\dataset\archive\colored_images'):
  # get the list of images in a given class
  train_class = os.listdir(os.path.join(r'C:\Users\shind\Downloads\codes\codes\dataset\archive\colored_images', i))
  # plot 5 images per class
  for j in range(5):
    img = os.path.join(r'C:\Users\shind\Downloads\codes\codes\dataset\archive\colored_images', i, train_class[j])
    img = PIL.Image.open(img)
    axs[count][j].title.set_text(i)
    axs[count][j].imshow(img)  
  count += 1

fig.tight_layout()


# In[20]:


# check the number of images in each class in the training dataset

No_images_per_class = []
Class_name = []
for i in os.listdir(r'C:\Users\shind\Downloads\codes\codes\dataset\archive\colored_images'):
  train_class = os.listdir(os.path.join(r'C:\Users\shind\Downloads\codes\codes\dataset\archive\colored_images', i))
  No_images_per_class.append(len(train_class))
  Class_name.append(i)
  print('Number of images in {} = {} \n'.format(i, len(train_class)))


# In[21]:


retina_df = pd.DataFrame({'Image': train,'Labels': label})
retina_df


# In[22]:


# Shuffle the data and split it into training and testing
retina_df = shuffle(retina_df)
train, test = train_test_split(retina_df, test_size = 0.2)


# In[23]:


# Define input shape and number of classes
input_shape = (256, 256, 3)
num_classes = 5


# In[24]:


# Create run-time augmentation on training and test dataset
# For training datagenerator, we add normalization, shear angle, zooming range and horizontal flip
train_datagen = ImageDataGenerator(
        rescale = 1./255,
        shear_range = 0.2,
        validation_split = 0.15)

# For test datagenerator, we only normalize the data.
test_datagen = ImageDataGenerator(rescale = 1./255)# Creating datagenerator for training, validation and test dataset.



# In[25]:


train_generator = train_datagen.flow_from_dataframe(
    train,
    directory='./',
    x_col="Image",
    y_col="Labels",
    target_size=(256, 256),
    color_mode="rgb",
    class_mode="categorical",
    batch_size=32,
    subset='training')

validation_generator = train_datagen.flow_from_dataframe(
    train,
    directory='./',
    x_col="Image",
    y_col="Labels",
    target_size=(256, 256),
    color_mode="rgb",
    class_mode="categorical",
    batch_size=32,
    subset='validation')

test_generator = test_datagen.flow_from_dataframe(
    test,
    directory='./',
    x_col="Image",
    y_col="Labels",
    target_size=(256, 256),
    color_mode="rgb",
    class_mode="categorical",
    batch_size=32)


# In[26]:


# Define the AlexNet model
model = Sequential([
    # 1st Convolutional Layer
    Conv2D(filters=96, input_shape=(256,256,3), kernel_size=(11,11), strides=(4,4), padding='valid', activation='relu'),
    MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'),
    BatchNormalization(),

    # 2nd Convolutional Layer
    Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), padding='same', activation='relu'),
    MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'),
    BatchNormalization(),

    # 3rd Convolutional Layer
    Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'),
    BatchNormalization(),

    # 4th Convolutional Layer
    Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'),
    BatchNormalization(),

    # 5th Convolutional Layer
    Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'),
    MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'),
    BatchNormalization(),

    # Flatten the CNN output to feed it into a Dense layer
    Flatten(),

    # 1st Dense Layer
    Dense(units=4096, input_shape=(224*224*3,), activation='relu'),
    Dropout(0.5),

    # 2nd Dense Layer
    Dense(units=4096, activation='relu'),
    Dropout(0.5),

    # 3rd Dense Layer (Output Layer)
    Dense(units=5, activation='softmax')
])


# In[27]:


# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[28]:


# Display the model summary
model.summary()


# In[29]:


#using early stopping to exit training if validation loss is not decreasing even after certain epochs (patience)
earlystopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)

#save the best model with lower validation lossmodel.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics= ['accuracy'])

checkpointer = ModelCheckpoint(filepath="weights.hdf5", verbose=1, save_best_only=True)


# In[30]:


history = model.fit(train_generator, steps_per_epoch = train_generator.n // 32, epochs = 18, validation_data= validation_generator, validation_steps= validation_generator.n // 32, callbacks=[checkpointer , earlystopping])


# In[31]:


# Evaluate the performance of the model
evaluate = model.evaluate(test_generator, steps = test_generator.n // 32, verbose =2)

print('Accuracy Test : {}'.format(evaluate[1]))


# In[32]:


# Assigning label names to the corresponding indexes
labels = {0: 'Mild', 1: 'Moderate', 2: 'No_DR', 3:'Proliferate_DR', 4: 'Severe'}


# In[33]:


# Loading images and their predictions 

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import cv2

prediction = []
original = []
image = []
count = 0

for item in range(len(test)):
  #code to open the image
  img= PIL.Image.open(test['Image'].tolist()[item])
  #resizing the image to (256,256)
  img = img.resize((256,256))
  #appending image to the image list
  image.append(img)
  #converting image to array
  img = np.asarray(img, dtype= np.float32)
  #normalizing the image
  img = img / 255
  #reshaping the image in to a 4D array
  img = img.reshape(-1,256,256,3)
  #making prediction of the model
  predict = model.predict(img)
  #getting the index corresponding to the highest value in the prediction
  predict = np.argmax(predict)
  #appending the predicted class to the list
  prediction.append(labels[predict])
  #appending original class to the list
  original.append(test['Labels'].tolist()[item])


# In[34]:


#Getting the test accuracy 
score = accuracy_score(original,prediction)
print("Test Accuracy : {}".format(score))


# In[35]:


# Visualizing the results
import random
fig=plt.figure(figsize = (100,100))
for i in range(20):
    j = random.randint(0,len(image))
    fig.add_subplot(20, 1, i+1)
    plt.xlabel("Prediction: " + prediction[j] +"   Original: " + original[j])
    plt.imshow(image[j])
fig.tight_layout()
plt.show()


# In[36]:


No_images_per_class
Class_name
fig1, ax1 = plt.subplots()
ax1.pie(No_images_per_class, labels = Class_name, autopct = '%1.1f%%')
plt.show



# In[37]:


tf.keras.preprocessing.image.ImageDataGenerator(
      featurewise_center=False,
      samplewise_center=False,
      featurewise_std_normalization=False,
      samplewise_std_normalization=False,
      zca_whitening=False,
      zca_epsilon=1e-06,
      rotation_range=0,
      width_shift_range=0.0,
      height_shift_range=0.0,
      brightness_range=None,
      shear_range=0.0,
      zoom_range=0.0,
      channel_shift_range=0.0,
      fill_mode="nearest",
      cval=0.0,
      horizontal_flip=False,
      vertical_flip=False,
      rescale=None,
      preprocessing_function=None,
      data_format=None,
      validation_split=0.3,
      dtype=None)


# In[38]:


# Print out the classification report
print(classification_report(np.asarray(original), np.asarray(prediction)))


# In[39]:


# plot the confusion matrix
plt.figure(figsize = (20,20))
cm = confusion_matrix(np.asarray(original), np.asarray(prediction))
ax = plt.subplot()
sns.heatmap(cm, annot = True, ax = ax)

ax.set_xlabel('Predicted')
ax.set_ylabel('Original')
ax.set_title('Confusion_matrix')


# In[ ]:



   

