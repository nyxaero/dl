# -*- coding: utf-8 -*-
"""
Created on Sun Dec  7 22:30:34 2025

@author: hemanth
"""


from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

(train_images,train_labels),(test_images, test_labels) = datasets.mnist.load_data()

train_images = train_images.reshape((train_images.shape[0],28,28,1))
test_images = test_images.reshape((test_images.shape[0],28,28,1))

train_images, test_images = train_images/255.0, test_images/255.0

model = models.Sequential([
       
       layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)),
       layers.MaxPooling2D((2,2)),
       
       layers.Conv2D(64,(3,3),activation='relu',input_shape=(28,28,1)),
       layers.MaxPooling2D((2,2)),
       
       layers.Conv2D(64,(3,3),activation="relu"),
       
       layers.Flatten(),
       
       layers.Dense(64,activation='relu'),
       
       layers.Dense(10,activation="softmax")
      
   ])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=5,validation_data=(test_images,test_labels))

test_loss , test_acc = model.evaluate(test_images,test_labels,verbose=2)





plt.figure(figsize=(12, 4))

# Accuracy subplot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='upper right')
plt.title('Training and Validation Accuracy')

# Loss subplot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss', marker='o')
plt.plot(history.history['val_loss'], label='Validation Loss', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.tight_layout()
plt.show()
