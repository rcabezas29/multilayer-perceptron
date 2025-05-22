# multilayer-perceptron

This project is an educational implementation of a **Multilayer Perceptron (MLP)** neural network from scratch. It is designed to classify whether a breast cancer diagnosis is malignant or benign, using the **Wisconsin Breast Cancer** dataset.

The multilayer perceptron is a feedforward network (meaning that the data
flows from the input layer to the output layer) defined by the presence of one or more
hidden layers, as well as an interconnection of all the neurons of one layer to the next.


![mlp](./assets/multilayer-perceptron.png)

The diagram above represents a network containing 4 dense layers (also called fully
connected layers). Its inputs consist of 4 neurons and its output consists of 2 (perfect for binary classification). The weights of one layer to the next are represented by two-dimensional matrices noted Wlj lj+1 . The matrix Wl0l1 is of size (3, 4) for example, as it contains the weights of the connections between layer l0 and layer l1.

The bias is often represented as a special neuron which has no inputs and an output
always equal to 1. Like a perceptron, it is connected to all the neurons of the following layer (the bias neurons are noted blj on the diagram above). The bias is generally useful as it allows one to “control the behavior” of a layer.

## 🧠 Perceptron

The perceptron is the type of neuron that the multilayer perceptron is composed
of. It is defined by the presence of one or more input connections, an activation
function, and a single output. Each connection contains a weight (also called a
parameter) which is learned during the training phase.

![mlp](./assets/perceptron.png)

## 🧮 Activation Functions

Activation functions introduce non-linearity into the neural network, allowing it to learn complex patterns. Below are the key activation functions used in this project:

### 1. Sigmoid Function

The **sigmoid** activation function maps any real-valued number to the range (0, 1). It’s useful for binary classification and is often used in output layers when the goal is to produce probabilities.

**Formula:**

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

**Properties:**
- Smooth, differentiable
- Output between 0 and 1
- Can suffer from vanishing gradients

---

### 2. ReLU (Rectified Linear Unit)

The **ReLU** function is the most widely used activation in hidden layers due to its simplicity and effectiveness. It allows only positive values to pass through.

**Formula:**

$$
\text{ReLU}(x) = \max(0, x)
$$


**Properties:**
- Fast to compute
- Avoids saturation in the positive domain
- Can suffer from "dying ReLUs" (neurons stuck at 0)

---

### 3. Softmax Function

The **softmax** function is typically used in the output layer for **multi-class** or **binary classification (2 neurons)**, as it converts raw logits into a probability distribution over multiple classes.

**Formula** (for output vector `z = [z₁, z₂, ..., zₖ]`):

$$
\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}
$$

**Properties:**
- Outputs sum to 1 (interpreted as probabilities)
- Amplifies differences between values
- Often paired with categorical cross-entropy loss

---

### 4. 📈 Tanh (Hyperbolic Tangent)

The **tanh** function is similar to sigmoid but maps input values to the range **(-1, 1)** instead of (0, 1). It is zero-centered, which can help optimization converge faster than sigmoid.

**Formula:**

$$
\tanh(x) = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}
$$

**Properties:**
- Output between -1 and 1
- Smooth and differentiable
- Zero-centered (unlike sigmoid)
- Can still suffer from vanishing gradients for large values

---

### 5. ⚡ Leaky ReLU

**Leaky ReLU** is a variation of ReLU that allows a small, non-zero gradient when the unit is not active (i.e., when $x < 0$). This addresses the "dying ReLU" problem.

**Formula:**

$$
\text{LeakyReLU}(x) = \begin{cases}
x & \text{if } x \geq 0 \\
\alpha x & \text{if } x < 0
\end{cases}
$$

Where $\alpha$ is a small constant (commonly $\alpha = 0.01$).

**Properties:**
- Avoids dying ReLU by allowing small negative outputs
- Introduces slight complexity with hyperparameter $\alpha$
- Often used as a default fallback to standard ReLU

---

### 6. 🔥 ELU (Exponential Linear Unit)

The **ELU** function improves upon ReLU and Leaky ReLU by allowing the negative part to saturate smoothly, which can help the network converge faster and perform better.

**Formula:**

$$
\text{ELU}(x) = \begin{cases}
x & \text{if } x \geq 0 \\
\alpha (e^x - 1) & \text{if } x < 0
\end{cases}
$$

Where $\alpha > 0$ controls the value to which an ELU saturates for negative inputs.

**Properties:**
- Smooth and continuous
- Negative values help center activations around zero
- Slightly more computationally expensive than ReLU

---

These functions are crucial for enabling the neural network to learn effectively and are chosen depending on the role of each layer in the network.

## 📉 Cross-Entropy Error Function

The **Cross-Entropy Error Function** is a widely used loss function for classification problems, especially when using softmax or sigmoid activations in the output layer. It measures the dissimilarity between the predicted probability distribution and the actual labels. The goal during training is to minimize this error.

For binary classification, the formula is:

$$
E = -\frac{1}{N} \sum_{n=1}^{N} \left[ y_n \log(p_n) + (1 - y_n) \log(1 - p_n) \right]
$$

Where:
- $N$ is the number of samples
- $y_n$ is the true label (0 or 1) for sample $n$
- $p_n$ is the predicted probability of the positive class for sample $n$

For multi-class classification with softmax, the categorical cross-entropy is used:

$$
E = - \sum_{i=1}^{K} y_i \log(p_i)
$$

Where:
- $K$ is the number of classes
- $y_i$ is a one-hot encoded label vector
- $p_i$ is the predicted probability for class $i$

Cross-entropy penalizes confident but incorrect predictions heavily, making it highly effective for classification tasks.

## ⚙️ Optimizers

Optimizers are algorithms used to minimize the loss function by updating the weights of the neural network during training. Choosing the right optimizer can significantly impact the model’s performance and convergence speed.

### 🧠 Adam Optimizer (Adaptive Moment Estimation)

**Adam** is an adaptive learning rate optimization algorithm that combines the advantages of two other methods: **AdaGrad** and **RMSProp**. It computes individual adaptive learning rates for each parameter by maintaining both the **first moment (mean)** and **second moment (uncentered variance)** of the gradients.

**Update equations:**

Given gradients $g_t$ at time step $t$:

1. Compute biased first moment estimate:
   $$
   m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t
   $$

2. Compute biased second raw moment estimate:
   $$
   v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2
   $$

3. Correct bias in the estimates:
   $$
   \hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
   $$

4. Update parameters:
   $$
   \theta_t = \theta_{t-1} - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
   $$

Where:
- $\eta$ is the learning rate
- $\beta_1$ and $\beta_2$ are decay rates (commonly 0.9 and 0.999)
- $\epsilon$ is a small number to avoid division by zero

**Advantages:**
- Works well with sparse gradients
- Requires minimal tuning
- Fast convergence

---

### ⏹️ Early Stopping

**Early Stopping** is a regularization technique used to prevent overfitting. It monitors the validation loss during training and halts training when the performance stops improving.

In this implementation, the training process stops if the validation loss reaches a mimimum. This helps preserve the best version of the model without overtraining it.

**How it works:**
- Track the best validation loss observed
- If the validation loss hasn't improved for `patience` epochs, stop training
- Optionally, restore the best model weights

**Benefits:**
- Saves training time
- Improves generalization
- Reduces risk of overfitting


## 🧬 [Dataset](./datasets/data.csv)

- Source: [UCI Machine Learning Repository](archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)

- Features: 30 numerical features representing characteristics of cell nuclei

- Label: M (malignant) or B (benign)

## Resources

