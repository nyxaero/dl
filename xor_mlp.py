import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

x = np.array([[0, 0], [0, 1], [1, 0], [1,1]]) 
y = np.array( [ [0],[1],[1],[0] ] )

model = Sequential([
        Dense(8, input_dim=2, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

model.compile(
        loss='binary_crossentropy',
        optimizer = SGD(learning_rate=0.1),
        metrics=['accuracy']
    )

model.fit( x, y, epochs = 1000, verbose=0 )

_, accuracy = model.evaluate(x , y)
print(accuracy)

predictions = model.predict(x)
print(predictions)

print(np.round(predictions))