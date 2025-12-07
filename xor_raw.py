import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return x*(1-x)

inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
expected_output = np.array([[0],[1],[1],[0]])

input_Layer_Neurons , hidden_Layer_Neurons, output_Layer_Neurons = 2,2,1

hidden_Layer_Weights = np.random.uniform(size=(input_Layer_Neurons,hidden_Layer_Neurons))
hidden_Layer_Bias = np.random.uniform(size=(1,hidden_Layer_Neurons))

output_Layer_Weights = np.random.uniform(size=(hidden_Layer_Neurons,output_Layer_Neurons))
output_Layer_Bias = np.random.uniform(size=(1,output_Layer_Neurons))

lr = 0.1

epochs = 10000
for _ in range(epochs):

    # forward prop 
    hidden_Layer_Activation = np.dot(inputs, hidden_Layer_Weights) + hidden_Layer_Bias
    hidden_Layer_Output = sigmoid(hidden_Layer_Activation)

    output_Layer_Activation = np.dot(hidden_Layer_Output, output_Layer_Weights) + output_Layer_Bias
    predicted_Output = sigmoid(output_Layer_Activation)

    error = expected_output - predicted_Output
    d_Output_Layer = error * sigmoid_derivative(predicted_Output)

    error_Hidden_Layer = d_Output_Layer.dot(output_Layer_Weights.T)
    d_hidden_layer = error_Hidden_Layer * sigmoid_derivative(hidden_Layer_Output)

    output_Layer_Weights += hidden_Layer_Output.T.dot(d_Output_Layer) * lr
    output_Layer_Bias += np.sum(d_Output_Layer, axis=0, keepdims=True) *  lr
    hidden_Layer_Weights += inputs.T.dot(d_hidden_layer) * lr
    hidden_Layer_Bias += np.sum(d_hidden_layer,  axis=0, keepdims=True)

print(predicted_Output)

