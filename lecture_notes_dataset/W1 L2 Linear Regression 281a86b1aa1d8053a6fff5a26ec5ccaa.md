# W1 L2: Linear Regression

3 October 2025 

[Week1Lec2LinearRegression.pdf](W1%20L2%20Linear%20Regression/Week1Lec2LinearRegression.pdf)

## Simple regression example

<aside>
📎

Linear graph, given three data points: (3,6); (2,4); (1,2). Predict the output of x = 5. 

- Propose a simple model family $y = wx$
    - easy to see that w = 2
</aside>

### Process for solving

![Screenshot 2025-10-03 at 10.19.50.png](W1%20L2%20Linear%20Regression/Screenshot_2025-10-03_at_10.19.50.png)

1. Suggest an initial model
2. Loss computation: how far is our predictions from the true value (tested using data samples)
3. Gradient computation: which direction should alter to w to minimise your loss. 
4. Parameter update and repeat 

![Screenshot 2025-10-03 at 10.23.52.png](W1%20L2%20Linear%20Regression/Screenshot_2025-10-03_at_10.23.52.png)

The model: $\hat{y}=\sum w_i x_i$.

- implied sum over i’s, represents the number of features you have as inputs
- Our predicted value using the model: $\hat{y}_j=\sum w_i x_{ij}$
    - Note that the output is not a vector.
    - This notation means we are calculating the predicted values for $(x)_1^j$, which are the input $x$ vectors for which we have the real outputs $y$.
        - e.g. $x_3 = (x_{13}, x_{23}, ..., x_{n3})^T$, the first index is the feature and the second represents which input point we are using.
        - $\hat{y}_3=\sum w_i x_{i3}= w_1 x_{13}+...+w_n x_{n3}$, expanded form of the sum.

> Loss model used is mean squared error.
> 
> - The squared distance between your model prediction and your actual data

**In our example:** we are working with only one feature,  $w$

- Gradient computation
    
    $$
    dw = \partial L/\partial w \\=\frac{\partial}{\partial w}(\frac{1}{N} \sum(wx_j-y_j)^2)
    \\ dw = \frac{1}{N}*2x_j (wx_j-y_j)
    $$
    
- weight update with an associated learning rate

$$
w_{t+1} = w_{t}-learning_{rate}*dw
$$

- note we have a minus since we are moving in the direction opposite to the gradient as we are descending towards the minimum.

## Implementing this in code

```bash
import numpy as np

**# Training Samples**
X = np.array([1,0, 2.0, 3.0, 4.0], dtype=np.float32)
Y = np.array([2.0, 4.0, 6.0, 8.0], dtype=np.float32)
```

### **Using Numpy**

```bash
**# Initialising your weights**
w = 0.0 

**# model prediction**
def forward(x):
	return w * x

**# loss = MSE**
def loss(y, y_predicted):
	return ((y_predicted - y)**2).mean() # mean calculactes the 1/N

**# gradient** 
# MSE = 1/N(w.x - y)**2 # defined w.x to mean dot product
# dJ/dw = 1/N 2x(w.x - y)

def gradient(x, y, y_predicted):
	return np.dot(2*x, y_predicted - y).mean()
```

```bash
**# implementing the functions**
print(f"Prediction before training:f(5) = {forward(5):.3f}")
# started at an arbitrary value of w = 0, f(5) = 0.000
	
for epoch in range(n_iters):
 # prediction = forward_pass
 y_pred = forward(X)
 
 **# loss**
 l = loss(Y, y_pred)
 
 **# gradient**
 dw = gradient(X,Y, y_pred)
 
 **# update weights**
 w -= learning_rate * dw
```

```bash
 if epoch%1 == 0:
 print(f'epoch = {epoch:.3f} w = {w:.3f} loss = {l:.8f}')
```

- The use of epoch%1 == 0, allows you an easy way to change the items you are printing
    - %2 prints every other iteration
- There is no correct number of iteration

### Using Keras

```bash
**# important libraries and functions**
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import SGD

**# X and Y as defined above**
n_samples, n_features = X.shape
X_test = np.array([[5]], dtype=np.float32)
input_size = n_features
output_size = n_features
```

- X.shape gives the shape of the data X: samples, features
- X_test, has double brackets to show it is a 2-dimensional array
    - np.array([5]) would have the shape (1,)

```bash
**# model selection**
model = Sequential([
 Input(shape=(input_size,)),
 Dense(units=output_size, name='linear_layer')
])
```

- Simplest way to build a neural network: has no branches, no loops or multiple inputs
    - Has one layer
- **Input**: tells what the shape of the input data is, in this case (1, ) vector
- **Dense:** every neuron in this layer is connected to every neuron in the previous layer. i.e. a fully connected layer. It implies a linear transformation → can be written in matrix form:
    - This allows us to generalise to $y = Wx + b$.
        - W = weight matrix, b = bias vector
    - **Units:** how many neurons (outputs) this layer has
    - **Name:** Allows you to store the name of the layer you have defined in a dictionary format

```bash
**# Aside: Hidden Layer**
Sequential([
    Dense(3, activation='relu', input_shape=(2,)),  # hidden layer
    Dense(1)                                        # output layer
])
```

- This method means you can definte the input_shape within the Dense layer.
- activation and input shape are examples of keyword arguments

```bash
# Print initial prediction before training
initial_prediction = model.predict(X_test,
verbose=0)

Dense(units=output_size, use_bias=False, name='linear_layer')
```

- verbose = 0 means that there is no ouput printed, 1 or 2 prints a progress bar for the prediction.
- The initial prediction is a random value and the bias is set to start at 0.
- Using the second line when defining the code makes it so that the model doesn’t consider any bias term.

```bash
**# Define loss, optimizer, and training parameters**
learning_rate = 0.1
num_epochs = 200

loss_fn = 'mse' # Mean Squared Error)

**# Stochastic Gradient Descent function**
optimizer_fn = SGD(learning_rate=learning_rate) #

**# Compile the model**
model.compile(optimizer=optimizer_fn, loss=loss_fn)
print("\nStarting training with Keras model.fit...")

**# model.fit executes the training loop**
history = model.fit(X, Y, epochs=n_iters, verbose=0)

# print loss every 10 epochs
print_interval=10
for i in range(0,num_epochs,print_interval):
	loss_at_epoch = history.history['loss'][i]
	
# Extract weights and bias from the trained model
# Weights are stored as [W,b] in the first layer
W, b = model.layers[0].get_weights()
print(f'epoch = {i}, loss = {loss_at_epoch:.4f}, W = {W[0][0]:.4f}, b = {b[0]:.4f}')

**# model.fit executes the training loop**
history = model.fit(X, Y, epochs=n_iters, verbose=0)

**# print final prediction**
final_prediction = model.predict(X_test, verbose = 0)
print(f'Preduction after training: f(5) = {final_prediction[0][0]:3f}')
```

- model.compile() tells keras how to train the model. (doesn’t start the training yet)
- model.fit() runs forward passes to compute predictions.
- verbose non-zero will give information about the loss

<aside>
💡

**Forward pass or forward propogation:** is the process of passing the input data through the model to compute the predicted output. 

</aside>

## Mathematical formalism

### Model

- $\text{Parameters: } \phi$
- $\text{Model: } f[\bold{x}, \bold{\phi}]$
- $\text{Training dataset of I pairs: }\{\bold{x_i}, \bold{y_i}\}_{i=1}^I$
    
    **Loss function**
    
    $$
    L[\phi, f[\bold{x}, \bold{\phi}], \{\bold{x_i}, \bold{y_i}\}_{i=1}^I]
    $$
    

![Screenshot 2025-10-20 at 07.37.46.png](W1%20L2%20Linear%20Regression/Screenshot_2025-10-20_at_07.37.46.png)

<aside>

Mean squared error:

$$
L[\phi] = \sum_{i=1}^{I}{(f[\bold{x_i}, \bold{\phi}]-y_i)^2}
$$

- We choose the squared value since your distance from the true value can be psotie or negative.
- We often take the mean of the squared loss, so that the no. of samples don’t change the loss to be able to compare models of different sample sizes.

Find the parameters that minimise the loss:

$$
\hat{\phi} = \argmin_{\phi}[L[\phi]]
$$

</aside>

![Screenshot 2025-10-20 at 07.37.56.png](W1%20L2%20Linear%20Regression/Screenshot_2025-10-20_at_07.37.56.png)

### Testing

- To test the model, run on a separate test dataset to see how well it generalises to new data.

> Generalisation: degree to which the model trained on some initial data first to your test data.
> 

### Issues with Linear regression

- Assuming linear data where in real life it is uncommon
- MSE is sensitive to outliers
- Model with too many features can become too comples
    - Fits to statistical anomalies in your training data
    - Hence **overfitting**: doesn’t generalise well to new data

### Aside: Convexity

- Convex data has the property of having a single global minimum which allows convergence to exist.
- We also do not have any local minima which would cause problems with gradient descent.
- For non-convex data, we would be dealing with a multi-layer neural network.