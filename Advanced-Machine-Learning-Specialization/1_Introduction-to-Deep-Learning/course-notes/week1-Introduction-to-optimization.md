 [Course intro](#course-intro)
- [Course intro](#course-intro)
- [Linear model as the simplest neural network](#linear-model-as-the-simplest-neural-network)
- [Regularization in machine learning](#regularization-in-machine-learning)
- [Stochastic methods for optimization](#stochastic-methods-for-optimization)

# Course intro

Agenda:

- Week 1: start with linear models
- Week 2: train the simplest neural network
- Week 3: feed images to the network
- Week 4: working with neural representations
- Week 5: feed test
- Week 6: mission with the boss - train an image captioning model(image -> text)

# Linear model as the simplest neural network

- supervised learning
- Regression and classification
- Linear model for regression
- Loss function
- Training a model

  $$
  L(w)=\frac{1}{\ell}\|X w-y\|^{2} \rightarrow \min _{w}
  $$
  
  solution:

  $$
  w=\left(X^{T} X\right)^{-1} X^{T} y
  $$

- softmax transform

  $$
  \begin{aligned} \sigma(z) &=\left(\frac{e^{z_{1}}}{\sum_{k=1}^{K} e^{z_{k}}}, \ldots, \frac{e^{z_{K}}}{\sum_{k=1}^{K} e^{z_{k}}}\right) \\ & \text { (softmax transform) } \end{aligned}
  $$

- croos entrypy (for binary)

  $$
  -\sum_{k=1}^{K}[y=k] \log \frac{e^{z_{k}}}{\sum_{j=1}^{K} e^{z_{j}}}=-\log \frac{e^{z_{y}}}{\sum_{j=1}^{K} e^{z_{j}}}
  $$

- cross entropy (for multiple classes)
  $$
  \begin{aligned} L(w, b) &=-\sum_{i=1}^{\ell} \sum_{k=1}^{K}\left[y_{i}=k\right] \log \frac{e^{w_{k}^{T} x_{i}}}{\sum_{j=1}^{K} e^{w_{j}^{T} x_{i}}} \\ &=-\sum_{i=1}^{\ell} \log \frac{e^{w_{j}^{T} x_{i}}}{\sum_{j=1}^{K} e^{w_{j}^{T} x_{i}}} \rightarrow \min _{w} \end{aligned}
  $$

- gradient descent

  $$
  \begin{array}{l}{\text { Optimization problem: } \quad L(w) \rightarrow \min } \\ {w^{0}-\text { initialization }} \\ {\nabla L\left(w^{0}\right)=\left(\frac{\partial L\left(w^{0}\right)}{\partial w_{1}}, \dots, \frac{\partial L\left(w^{0}\right)}{\partial w_{n}}\right)-\text { gradient vector }} \\ {w^{1}=w^{0}-\eta_{1} \nabla L\left(w^{0}\right)-\text { gradient step }}\end{array}
  $$

- gradient descent heuristics
  - how to initialize weights
  - how to select learning rate
  - when to stop
  - how to approximate gradient $\nabla L\left(w^{t-1}\right)$

- MSE gradient descent

  $$
  \nabla L_{w}(w)=\frac{2}{\ell} X^{T}(X w-y)
  $$

# Regularization in machine learning

- Generalization
  - consider a model with accuracy 80% on training set
  - How it will perform on the data
  - Dose the model generalize well

- Holdout set: training set and holdout set
  - small holdout set: training set is representative, holdout quality has high variance
  - large holdout setL holdout quality has low variance, holdout quanlity has high bias

- cross-validation
  - requires to train models K times for K-fold CV
  - useful for samll samples
  - in deep dealrning holdout samples are usually preferred

- weight penalty

  $$
  L_{r e g}(w)=L(w)+\lambda\|w\|^{2} \rightarrow \min _{w}
  $$

  - L2 penalty
  - L1 penalty
    - dirves some weights exactly to zero
    - learns sparse models
    - cannot be optimized with simple gradient methods

- other regularization techniques
  - dimensionalizty reduction
  - data augmentation
  - dropout
  - ealry stopping
  - collect more data

# Stochastic methods for optimization

- stochastic gradient descent
  - noisy updates lead to fluctuations
  - needs only one example on each step
  - can be used in online setting
  - learning rate $\eta_{t}$ should be choosn very carefully

- mini-batch gradient descent
  - still can be used in online setting
  - reduces the variance of gradient approximations
  - learning rate $\eta_{t}$ should be choosn very carefully

- Momentum

  $$
  \begin{array}{c}{h_{t}=\alpha h_{t-1}+\eta_{t} g_{t}} \\ {w^{t}=w^{t-1}-h_{t}}\ \\ \text{usually  } \alpha = 0.9 \\ \end{array}
  $$

- Nesterov momentum

  $$
  \begin{array}{c}{h_{t}=\alpha h_{t-1}+\eta_{t} \nabla L\left(w^{t-1}-\alpha h_{t-1}\right)} \\ {w^{t}=w^{t-1}-h_{t}}\end{array}
  $$

- AdaGrad

  $$
  \begin{array}{c}{G_{j}^{t}=G_{j}^{t-1}+g_{t j}^{2}} \\ {w_{j}^{t}=w_{j}^{t-1}-\frac{\eta_{t}}{\sqrt{G_{j}^{t}+\epsilon}} g_{t j}}\end{array}
  $$

  - $g_{t j}$ - gradient with respect to j-th parameter
  - separate learning rates for each dimentsion
  - suits for sparse data
  - learning rate can be fixed: $\eta_{t}=0.01$
  - $G_{j}^{t}$ always increases, leads to early stops

- RMSprop

  $$
  \begin{aligned} G_{j}^{t} &=\alpha G_{j}^{t-1}+(1-\alpha) g_{t j}^{2} \\ w_{j}^{t} &=w_{j}^{t-1}-\frac{\eta_{t}}{\sqrt{G_{j}^{t}+\epsilon}} g_{t j} \end{aligned}
  $$

  - $\alpha$ is about 0.9
  - learning rate adapts to latest gradient steps

- Adam

  $$
  \begin{array}{l}{v_{j}^{t}=\frac{\beta_{2} v_{j}^{t-1}+\left(1-\beta_{2}\right) g_{t j}^{2}}{1-\beta_{2}^{t}}} \\ {w_{j}^{t}=w_{j}^{t-1}-\frac{\eta_{t}}{\sqrt{v_{j}^{t}}+\epsilon} g_{t j}}\end{array}
  $$

  combines momentum and individual learnign rates

  $$
  \begin{array}{l}{m_{j}^{t}=\frac{\beta_{1} m_{j}^{t-1}+\left(1-\beta_{1}\right) g_{t j}}{1-\beta_{1}^{t}}} \\ {v_{j}^{t}=\frac{\beta_{2} v_{j}^{t-1}+\left(1-\beta_{2}\right) g_{t j}^{2}}{1-\beta_{2}^{t}}} \\ {w_{j}^{t}=w_{j}^{t-1}-\frac{\eta_{t}}{\sqrt{v_{j}^{t}+\epsilon}} m_{j}^{t}}\end{array}
  $$

  
