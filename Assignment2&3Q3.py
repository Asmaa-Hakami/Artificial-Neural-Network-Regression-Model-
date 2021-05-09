import pandas as pd
import numpy as np

# Define the sigmoid function
def sigmoid(x):
    return 1/(1+np.exp(-x))

# Derivative of sigmoid function:
def sigmoid_der(x):
    return sigmoid(x)*(1-sigmoid(x))

# Main logic for neural network:
def NT_train (b1, b2, x, y):
    # Initialize the parameters 
    W_hidden = np.array([[0.1,0.4,0.3], [0.2,0.3,0.5], [0.4,0.2,0.3]]) #np.random.rand(len(x[0]),3)
    W_output = np.array([[0.4], [0.45], [0.3]]) #np.random.rand(3, 1)
    lr = 0.5
    
    for epoch in range(len(x)):
        # ======================= feedforward =======================
        t = [] #input for each hidden layer
        h = [] # 3 hidden layer in this model
    
        for i in range (len(W_hidden)):
            t.append(np.dot(x[epoch], W_hidden[i]) + b1) # t for each hidden layer
            h.append(sigmoid(t[i]))
        
        s = np.dot(h, W_output) + b2 #input for output layer
        y_pred = s #with no activation function
        
        print("input of hidden layers: \n", t) #input of hidden layers
        print("output of hidden layers: \n",h) #output of hidden layers
        print("input of output layer: \n",s) #input of output layer 
        print("output of output layer: \n",y_pred) #output of output layer

        # ======================= back-propagation =======================
        error = y_pred - y[epoch]

        #sum squared error
        SSE =+ np.power(error, 2)
        print("SSE is ",SSE)
    
        s_der = sigmoid_der(s) 
        dzo_dwo = np.array(h)
        dzo_dwo = dzo_dwo.reshape(3,1)
        dcost_wo = np.dot(dzo_dwo, SSE * s_der) # using sum square error (SSE).

        # ======================= back-propagation for hidden neurons =======================
        dcost_s = SSE * s_der
        dzo_dah = W_output.reshape(1,3)
        dcost_dah = np.dot(dcost_s , dzo_dah)
        dah_dzh = []
        for i in range(len(t)):
            dah_dzh.append(sigmoid_der(t[i]))

        dzh_dwh = x[epoch]
        dcost_wh = np.dot(dzh_dwh, np.dot(dah_dzh, dcost_dah))
        
        # ======================= Update Weights =======================
        W_hidden -= lr * dcost_wh
        dcost_wo = dcost_wo.reshape(3,1)
        W_output -= lr * dcost_wo

        print("Weight matric for hidden\n",W_hidden)
        print("Weight matric for output\n",W_output)
    return W_hidden, W_output


X_train = np.array([[300,99,6.8],[308,103,8.36],[329,110,9.15],[332,118,9.36]])
Y_train = np.array([[0.36 ,0.7 ,0.84 ,0.9]])
Y_train = Y_train.reshape(4,1)

b1 =0.6
b2 = 0.3
W_hidden, W_output = NT_train(b1,b2,X_train, Y_train)

X_test = np.array([[296,95,7.54],[293,97,7.8],[325,112,8.96]])
Y_test = np.array([[0.44 ,0.64 ,0.8]])
Y_test = Y_test.reshape(3,1)

y_pred = []

for epoch in range(len(X_test)):
        
    t = [] #input for each hidden layer
    h = [] # 3 hidden layer in this model
        
    for i in range (len(W_hidden)):
        t.append(np.dot(X_test[epoch], W_hidden[i]) + b1) # t for each hidden layer
        h.append(sigmoid(t[i]))

    h_aaray = np.array(h)
    h_aaray = h_aaray.reshape(3,1)
    s = np.dot(h, W_output) + b2 #input for output layer
    y_pred.append(s)
    #sum squared error
    SSE =+ np.power(y_pred[epoch] - Y_test[epoch], 2)

print(y_pred)
print("SSE is ",SSE)

