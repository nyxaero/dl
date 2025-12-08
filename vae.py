import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import mnist


(X_train, y_train), (X_test, y_test) = mnist.load_data()

print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")


plt.subplot(221)
plt.imshow(X_train[0], cmap='gray')

plt.subplot(222)
plt.imshow(X_train[1], cmap='gray')

plt.subplot(223)
plt.imshow(X_train[2], cmap='gray')

plt.subplot(224)
plt.imshow(X_train[3], cmap='gray')

plt.show()





num_pixels = X_train.shape[1] * X_train.shape[2]


X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test  = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')

X_train = X_train / 255.0
X_test  = X_test / 255.0

print(f"Shape of reshaped training data: {X_train.shape}")
print(f"Shape of reshaped test data: {X_test.shape}")



noise_factor = 0.2


X_train_noisy = X_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_train.shape)
X_test_noisy  = X_test  + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_test.shape)

X_train_noisy = np.clip(X_train_noisy, 0.0, 1.0)
X_test_noisy  = np.clip(X_test_noisy, 0.0, 1.0)



model = Sequential()

# Encoder
model.add(Dense(300, input_dim=num_pixels, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(50, activation='relu'))

# Decoder
model.add(Dense(784, activation='sigmoid'))  



model.compile(loss='mean_squared_error',
              optimizer='adam')


model.fit(X_train_noisy, X_train,     
          validation_data=(X_test_noisy, X_test),
          epochs=10,
          batch_size=256,
          verbose=1)


print("Evaluating")
pred = model.predict(X_test_noisy)

print(f"Shape of predicted data: {pred.shape}")
print(f"Shape of test data: {X_test.shape}")




X_test = X_test.reshape(-1, 28, 28) * 255
pred = pred.reshape(-1, 28, 28) * 255
X_test_noisy = X_test_noisy.reshape(-1, 28, 28) * 255




plt.figure(figsize=(10, 4))
print("Original Test Images")
for i in range(10):
    plt.subplot(2, 10, i + 1)
    plt.imshow(X_test[i], cmap='gray')
    plt.axis('off')
plt.show()


plt.figure(figsize=(10, 4))
print("Test Images with Noise")
for i in range(10):
    plt.subplot(2, 10, i + 1)
    plt.imshow(X_test_noisy[i], cmap='gray')
    plt.axis('off')
plt.show()


plt.figure(figsize=(10, 4))
print("Reconstruction of Noisy Test Images")
for i in range(10):
    plt.subplot(2, 10, i + 1)
    plt.imshow(pred[i], cmap='gray')
    plt.axis('off')
plt.show()


