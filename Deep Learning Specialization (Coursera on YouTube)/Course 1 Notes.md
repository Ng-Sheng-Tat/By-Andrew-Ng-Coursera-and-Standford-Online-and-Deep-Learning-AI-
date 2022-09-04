# [Link](https://www.youtube.com/playlist?list=PLpFsSf5Dm-pd5d3rjNtIXUHT-v7bdaEIe)

# Course 1: Neural Networks and Deep Learning

## Table of Content
1. 
2. 
3. 

## Week 1: Introduction
### What is a neural network?
- feeding input into a single neurons, often draw with a circle to map input to output
- *ReLu* function
<figure>

<img src = "https://marskar.github.io/frm/images/relu.png" alt = "ReLu Function">

<figcaption align = "center"><b>ReLu Function</b></figcaption>

</figure>
- single neuron network piles up to become a deep neural networks that many neurons are interconnected to each other

### Supervised Learning with a Neural Network
**Typical Neural Network**
1. Standard Neural Network
2. Convolutional Neural Network (CNN): for images
3. Recurrent Neural Network (RNN): for sequence data/time series
4. Custom or Hybrid Neural Network

### Why is deep learning taking off?
- Algorithm starts to reach plantau in the graph of performance against the amount of data used for trainining
- a large neural network is the secreat to reach a bigher level of performance

**Scale that drives deep learning progress**
- data has grown exponentially
- computation resources have increased
    - helps in the cycle of ideas -> code -> experiment -> ideas
- improvement of algorithm that saves computational resources
    - e.g.: changing the sigmoid function to ReLu function allows gradient descent algorithm to run faster and converges

## Week 2: Basics of Neural Network Programming

### Binary Classifications
**Notation**
- y $\epsilon$ {0, 1}
- x $\epsilon IR ^{n_x}$
- m training examples: m<sub>train</sub>, m<sub>test</sub> -> {(x<sup>(1)</sup>, y<sup>(1)</sup>)...(x<sup>(m)</sup>, y<sup>(m)</sup>)}
- x is a matrix of size n<sub>x</sub> by m, where each column is x<sup>(i)</sup>
- y is a matrix of size 1 by m, and it is [y<sup>(i)</sup> ... y<sup>m</sup>]

### Logistic Regressions
- output: $\hat{y} = P(y=1 | x)$, and $0 <= \hat{y} <= 1$
- where $w \epsilon IR^{n_x}$, and $b \epsilon IR$
- output: $hat{y} = \sigma (w^Tx + b)$
- where $w^Tx + b$ can be denoted as z and passing it into a sigmoid function
- $\sigma{(z)} = \frac{1}{1+e^{-z}}$
<figure>

<img src = "https://marskar.github.io/frm/images/relu.png" alt = "ReLu Function">

<figcaption align = "center"><b>ReLu Function</b></figcaption>

</figure>

- alternative notation
- given $x_o = 1$, and $x \epsilon IR^{(n_x + 1)}$, and $\hat{y} = \sigma{(\theta ^Tx)}$
- and $\theta$ is a vector of [$\theta_o, ... \theta n_x$], and $\theta _o$ is b

<figure>

<img src = "https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/Logistic-curve.svg/1200px-Logistic-curve.svg.png" alt = "Sigmoid Function">

<figcaption align = "center"><b>Sigmoid Function</b></figcaption>

</figure>

### Cost Function
**Loss Function**
- common one $\mathcal{L}(\hat{y}, y)=\frac{1}{2}(\hat{y}-y)^2$ does not allow one to find the global minimum
- since the output of y is only 0 or 1
- $\mathcal{L}(\hat{y}, y)=-(y \log{\hat{y}} + (1-y)\log(1-\hat{y}))$
- [Loss Function](https://www.desmos.com/calculator/fvvhigyqd6): the goal is to have the y-axis as the loss value to minimum among the range of x from 0 to 1
- Loss function is done on each example
- Cost function: $J(w, b) = \frac{1}{m} \sum_{i=1}^m \mathcal{L}(\hat{y}^(i), y^(i))$

### Gradient Descent
- an algorithm to fund w, and b that minimize $J(w, b)$
- goes by iteration to reach the global minimum/optimum
- imagine that the curve $J(w)$ against $w$ is a concave shape quadratic equation
- repeat: $w := w - \alpha \frac{\partial{(J(w, b)}}{\partial{w}} $ and $b := b - \alpha \frac{\partial{J(w, b)}}{\partial{b}} $, where $\frac{\partial{(J(w, b)}}{\partial{w}} $ may be simplified as $dw$ and $\frac{\partial{J(w, b)}}{\partial{b}}$ can be simplified as $db$
- need to know the elegant design of "-" instead of possitive

### Computation Graph: as a step by step substitution to find model output
- Forward pass: compute the output of the model
- Backward pass: compute the derivatives or gradient
- usually you need to find the derivative of some output variable that you care about to minimize and take many of it partial derivatives to all the variables, and applies chain rule for your calculations, that is $\frac{doutput var}{dvars}$, the naming convention in python will be ``dvar``




