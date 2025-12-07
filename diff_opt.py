
from tensorflow.keras import layers, models, optimizers
import numpy as np

import matplotlib.pyplot as plt


def create_data():
    x=np.random.randn(1000,10)
    y=np.random.randn(1000,1)
    return x,y

def create_model():
    model = models.Sequential([
        layers.Dense(50,activation="relu", input_shape=(10,)),
        layers.Dense(20,activation="relu"),
        layers.Dense(1)
    ])
    return model


def train_model_with_history(model, optimizer, x,y,batch_size,epochs,optimizer_name):
    model.compile(
        optimizer = optimizer, loss='mean_squared_error'
    )
    history=[]
    
    for epoch in range(epochs):
        hist = model.fit(x,y,batch_size=batch_size,epochs=1, verbose=0)
        loss = hist.history['loss'][0]
        history.append(loss)
        
        print(f"epoch {epoch+1}/{epochs} - {optimizer_name} loss: {loss}")
        
    return history

x,y = create_data()

model_sgd = create_model()
model_adam = create_model()

optimizer_sgd = optimizers.SGD(learning_rate=0.01)
optimizer_adam = optimizers.Adam(learning_rate=0.001)

epochs = 50

batch_size = 32

sgd_loss = train_model_with_history(model_sgd, optimizer_sgd, x, y, batch_size, epochs, 'sgd')
adam_loss = train_model_with_history(model_adam, optimizer_adam, x, y, batch_size, epochs, "adam")


plt.plot(range(1,epochs+1), sgd_loss, label='sgd',color='red')
plt.plot(range(1,epochs+1), adam_loss, label='sgd',color='blue')

plt.xlabel("epochs")
plt.ylabel("loss")
plt.title("sgd vs adam optimizer loss comparision")
plt.legend()
plt.grid(True)
plt.show()


