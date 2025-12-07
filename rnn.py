

from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

(x_train, y_train) , (x_test, y_test) = mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

x_train = x_train.reshape(
        ( x_train.shape[0],x_train.shape[1],x_train.shape[2] )
    )

x_test = x_test.reshape(
        (x_test.shape[0],x_test.shape[1],x_test.shape[2])
    )

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)


model = models.Sequential([
    layers.LSTM(128, activation='relu', input_shape=(x_train.shape[1] , x_train.shape[2])) ,
    layers.Dense(10, activation= 'softmax')
])


model.compile(
        optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']
    )


model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test,y_test))

test_loss, test_acc = model.evaluate(x_test, y_test)
