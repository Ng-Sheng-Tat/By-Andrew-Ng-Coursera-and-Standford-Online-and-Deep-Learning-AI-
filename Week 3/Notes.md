### Week 3 Notes

<fig>
    <center>
    <img src = "Deep Learning Frameworks.png" alt="Deep Learning Frameworks">
    <figcaption>Deep Learning Frameworks</figcaption>
</fig>

**Motivating Problem**
Cost function: $J(w) = w - 10w + 25$, which can be reduced to $J(w) = (w-5)^2$, and $w = 5$ after minimized. In real life, the cost function will be in $J(w, b)$



**Implementation of Tensorflow**

```
import numpy as np
import tensorflow as tf

w = tf.Variable(0, dtype=np.float32)

# setting alpha learning rate to 0.1
optimizer = tf.keras.optimizers.Adam(0.1)

def train_step():
    
    # gradient tape will keep track of the order of the iteration of the forward and backward propagation
    # with tensorflow, only forward propagation need to be considered, the backward (computation of the derivatives, gradient computation) propagation will be caterred by the deep learning framework library

    with tf.GradientTape() as tape:
        cost = w **2 - 10 * w + 25
    
    trainable_variables = [w]
    grads = tape.gradient(cost, trainable_variables)

    # zip function is the pairing of the gradient and the variables
    optimizer.apply_gradients(zip(grads, trainable_variables))

print(w)

for i in range(1000):
    train_step()

print(w)
```

**What if the x changes also**
```
# Tensorflow is very powerful that you only needs to specify the cost function to compute the parametesr
w = tf.Variable(0, dtype = tf.float32)
# serve as coefficient of the functions
x = np.array([1.0, -10.0, 25.0], dtype = np.float32)
optimizer = tf.keras.optimizers.Adam(0.1)

def training(x, w, optimizer):
    def cost_fn():
        return x[0] * w **2 + x[1] * w + x[2]
    
    for i in range(1000):
        optimizer.minimize(cost_fn, [w])
    
    return w

w = training(x, w, optimizer)

print(w)

```

<fig>
    <center>
    <img src = "Code Eg with Computation Graph.png" alt="Code Eg with Computation Graph">
    <figcaption>Code Eg with Computation Graph</figcaption>
</fig>

