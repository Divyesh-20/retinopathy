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
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
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


# Import necessary packages
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.initializers import glorot_uniform
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import PIL
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


from jupyterthemes import jtplot
jtplot.style(theme='grade3', context='notebook', ticks=True, grid=False) 
# setting the style of the notebook to be monokai theme  
# this line of code is important to ensure that we are able to see the x and y axes clearly
# If you don't run this code line, you will notice that the xlabel and ylabel on any plot is black on black and it will be hard to see them. 


# In[4]:


os.listdir(r'C:\Users\shind\Downloads\codes\codes\dataset\archive\colored_images')


# In[5]:


os.listdir(os.path.join(r'C:\Users\shind\Downloads\codes\codes\dataset\archive\train.csv', r'C:\Users\shind\Downloads\codes\codes\dataset\archive\colored_images\Mild'))


# In[7]:


# Check the number of images in the dataset
train = []
label = []


# In[8]:


# os.listdir returns the list of files in the folder, in this case image class names
for i in os.listdir(r'C:\Users\shind\Downloads\codes\codes\dataset\archive\colored_images'):
  train_class = os.listdir(os.path.join(r'C:\Users\shind\Downloads\codes\codes\dataset\archive\colored_images', i))
  for j in train_class:
    img = os.path.join(r'C:\Users\shind\Downloads\codes\codes\dataset\archive\colored_images', i, j)
    train.append(img)
    label.append(i)

print('Number of train images : {} \n'.format(len(train)))


# In[9]:


train


# In[10]:


label


# In[11]:


# Visualize 5 images for each class in the dataset

fig, axs = plt.subplots(5, 5, figsize = (20, 20))
count = 0
for i in os.listdir(r'C:\Users\shind\Downloads\codes\codes\dataset\archive\colored_images'):
  # get the list of images in a given class
  train_class = os.listdir(os.path.join(r'C:\Users\shind\Downloads\codes\codes\dataset\archive\colored_images',i))
  for j in range(5):
    img = os.path.join(r'C:\Users\shind\Downloads\codes\codes\dataset\archive\colored_images', i, train_class[j])
    img = PIL.Image.open(img)
    axs[count][j].title.set_text(i)
    axs[count][j].imshow(img)  
  count += 1

fig.tight_layout()


# In[12]:


# check the number of images in each class in the training dataset

No_images_per_class = []
Class_name = []
for i in os.listdir(r'C:\Users\shind\Downloads\codes\codes\dataset\archive\colored_images'):
  train_class = os.listdir(os.path.join(r'C:\Users\shind\Downloads\codes\codes\dataset\archive\colored_images', i))
  No_images_per_class.append(len(train_class))
  Class_name.append(i)
  print('Number of images in {} = {} \n'.format(i, len(train_class)))


# In[13]:


retina_df = pd.DataFrame({'Image': train,'Labels': label})
retina_df


# In[14]:


# Shuffle the data and split it into training and testing
retina_df = shuffle(retina_df)
train, test = train_test_split(retina_df, test_size = 0.2)


# In[15]:


# Create run-time augmentation on training and test dataset
# For training datagenerator, we add normalization, shear angle, zooming range and horizontal flip
train_datagen = ImageDataGenerator(
        rescale = 1./255,
        shear_range = 0.2,
        validation_split = 0.15)

# For test datagenerator, we only normalize the data.
test_datagen = ImageDataGenerator(rescale = 1./255)# Creating datagenerator for training, validation and test dataset.



# In[16]:


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


# In[17]:


# Function to create the hybrid model of CNN (InceptionResNetV2) and LSTM
def create_cnn_lstm_model(input_shape, num_classes):
    # Load InceptionResNetV2 model pre-trained on ImageNet
    base_model = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=input_shape)

    # Freeze the layers of InceptionResNetV2
    for layer in base_model.layers:
        layer.trainable = False

    # Extract CNN features
    cnn_features = base_model.output
    cnn_features = GlobalAveragePooling2D()(cnn_features)

    # Reshape features for LSTM
    lstm_input = Reshape((1, -1))(cnn_features)

    # LSTM layer
    lstm_output = LSTM(128)(lstm_input)

    # Fully connected layer
    fc_output = Dense(num_classes, activation='softmax', name='Dense_final')(lstm_output)

    # Combined model
    hybrid_model = Model(inputs=base_model.input, outputs=fc_output, name='HybridModel')
    return hybrid_model


# In[23]:


# Create the hybrid model
incep_lstm = create_cnn_lstm_model(input_shape=(256, 256, 3), num_classes=5)


# In[17]:





# In[18]:





# In[19]:





# In[24]:


# Compile the model
incep_lstm.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Display the summary of the model
incep_lstm.summary()


# In[45]:


#using early stopping to exit training if validation loss is not decreasing even after certain epochs (patience)
earlystopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)

#save the best model with lower validation lossmodel.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics= ['accuracy'])

checkpointer = ModelCheckpoint(filepath="weights.hdf5", verbose=1, save_best_only=True)


# In[ ]:





# In[26]:


# history = hybrid_model.fit(train_generator, steps_per_epoch = train_generator.n // 32, epochs = 18, validation_data= validation_generator, validation_steps= validation_generator.n // 32, callbacks=[checkpointer , earlystopping])
history = incep_lstm.fit(train_generator, steps_per_epoch = train_generator.n // 32, epochs = 50, validation_data= validation_generator, validation_steps= validation_generator.n // 32)


# In[27]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train_loss','val_loss'], loc = 'upper right')
plt.show()


# In[28]:


hybrid_model.load_weights("weights.hdf5")


# In[29]:


# Evaluate the performance of the model
evaluate = hybrid_model.evaluate(test_generator, steps = test_generator.n // 32, verbose =1)

print('Accuracy Test : {}'.format(evaluate[1]))


# In[30]:


labels = {0: 'No_DR', 1: 'Mild', 2: 'Moderate', 3:'Severe', 4: 'Proliferate_DR'}


# In[31]:


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
  predict = hybrid_model.predict(img)
  #getting the index corresponding to the highest value in the prediction
  predict = np.argmax(predict)
  #appending the predicted class to the list
  prediction.append(labels[predict])
  #appending original class to the list
  original.append(test['Labels'].tolist()[item])


# In[47]:


#Getting the test accuracy 
score = accuracy_score(original,prediction)
print("Test Accuracy : {}".format(score))


# In[48]:


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


# In[49]:


No_images_per_class
Class_name
fig1, ax1 = plt.subplots()
ax1.pie(No_images_per_class, labels = Class_name, autopct = '%1.1f%%')
plt.show


# In[50]:


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


# In[51]:


# Print out the classification report
print(classification_report(np.asarray(original), np.asarray(prediction)))


# In[52]:


# plot the confusion matrix
plt.figure(figsize = (20,20))
cm = confusion_matrix(np.asarray(original), np.asarray(prediction))
ax = plt.subplot()
sns.heatmap(cm, annot = True, ax = ax)

ax.set_xlabel('Predicted')
ax.set_ylabel('Original')
ax.set_title('Confusion_matrix')


# In[53]:


import numpy as np
from tensorflow.keras.utils import to_categorical

# Assuming you have a list of labels, for example:
label1 = label  # Replace this with your actual labels

# Convert string labels to integer indices
class_mapping = {label: idx for idx, label in enumerate(np.unique(label1))}
label_indices = [class_mapping[label] for label in label1]

# Convert labels to one-hot encoding
num_classes = len(np.unique(label1))
y_test = to_categorical(label_indices, num_classes=num_classes)

print(y_test)


# In[54]:


y_pred = hybrid_model.predict(test_generator)  # X_test should contain your test data
y_true = np.argmax(y_test, axis=1)  # y_test should contain true labels in one-hot encoded format


# In[55]:


from sklearn.metrics import confusion_matrix

# Assuming y_true is your true labels and y_pred is your predicted probabilities or class predictions
# If y_pred is probabilities, convert it to class predictions
y_pred_classes = np.argmax(y_pred, axis=1)

# Ensure that y_true and y_pred have the same length
y_true = y_true[:len(y_pred_classes)]

# Calculate confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)

# Calculate sensitivity and specificity
sensitivity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])

print(f"Sensitivity: {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")


# In[56]:


# Extract training history
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

# Plot Training and Validation Loss
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot Training and Validation Accuracy
plt.subplot(1, 2, 2)
plt.plot(train_acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()


# In[57]:


# Access training and validation losses from the history object
training_loss = history.history['loss']
validation_loss = history.history['val_loss']

# Print or plot the losses
print("Training Losses:", training_loss)
print("Validation Losses:", validation_loss)

# Plot the losses over epochs
plt.plot(training_loss, label='Training Loss')
plt.plot(validation_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[58]:


# Access training and validation losses from the history object
training_loss = history.history['loss']
validation_loss = history.history['val_loss']

# Print or plot the losses
print("Training Losses:", training_loss)
print("Validation Losses:", validation_loss)

# Plot the losses over epochs
plt.plot(training_loss, label='Training Loss')
plt.plot(validation_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[59]:


incep_lstm.save('incep_lstm.hdf5')


# In[ ]:




