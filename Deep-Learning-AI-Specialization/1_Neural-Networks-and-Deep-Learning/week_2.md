# week 2

## Logistic Regression as a Neural Network
- Basics of Neural Network Programming<br>
  Binary Classification<br>
  - Notation<br>
  $x \in \reals^n, y \in \{0, 1 \}$ <br>
  And there we have $m$ training samples.<br> 
  $X \in \reals^{n \cdot m}$<br>
  $Y \in \reals^{1 \cdot m}$
- Logistic Regression<br>
  $x \in \reals^n$<br>
  Parameters: $w \in \Reals^n, b \in \reals$<br>
  Output $\hat{y} = \sigma(w^T \cdot x + b$)<br>
  $\sigma(x) = \frac{1}{1 + e^{-x}}$

- Logistic Regression
  - lost function (with respect to a single training example)<br>
  
  $L(\hat{y}, y) = -\big(y\log \hat{y} + (1-y)\log(1- \hat{y})\big)$<br>

  The reason not using squared error is non-convex
  - cost function (entire training set)<br>
  
  $J(w,b) = \frac{1}{m}\sum^m_{i=1}L(\hat{y}^{(i)}, y^{(i)})$

- Gradient Descent<br>
  We want $argmin_{w,b}\big(J(w,b)\big)$<br>
  Repeat {
      $w := w - \alpha \frac{\partial{J(w,b)}}{\partial{w}}$<br>
      $b := b - \alpha \frac{\partial{J(w,b)}}{\partial{b}}$
  }<br>

- Derivatives<br>
  $\frac{\partial{\ln{a}}}{\partial a} = \frac{1}{a}$

- Computation Graph - back propogation computing derivatives
  
  $\frac{\partial{J}}{\partial{a}} = \frac{\partial{J}}{\partial{v}} \cdot \frac{\partial{v}}{\partial{a}}$

  we assume that $J = 3 \cdot v, v = (a + u)$

  For convenience:

  $\frac{\partial{FinalOutputVar}}{\partial{Var}} = \partial{Var}$

  So $\frac{\partial{J}}{\partial{v}} = \partial{v}, \frac{\partial{J}}{\partial{a}} = \partial{a}, \frac{\partial{J}}{\partial{u}} = \partial{u}$

- Logistic regression lost function derivative
  
  In a computing graph:

  $x_1, w_1, x_2, w_2, b \rightarrow Z = w_1 x_1 + w_2 x_2 + b \rightarrow \hat{y} = \sigma(Z) \rightarrow L(\hat{y}, y)$

  - Step 1:
  
    $\partial{\hat{y}} = \frac{\partial{L(\hat{y}, y)}}{\partial{\hat{y}}} = - \frac{y}{\hat{y}} + \frac{1-y}{1-\hat{y}}$

  - Step 2:
  
    $\partial{Z} = \frac{\partial{L}}{\partial{Z}} = \hat{y} - y = \frac{\partial{L}}{\partial{\hat{y}}} \frac{\hat{y}}{Z}, where \frac{\partial{\hat{y}}}{\partial{Z}} = \hat{y}(1-\hat{y})$
  
  - Step 3:

    $\partial{w_1} = \frac{\partial{L}}{\partial{w_1}} = x_1 \cdot \partial{Z}$

    $\partial{w_2} = \frac{\partial{L}}{\partial{w_2}} = x_2 \cdot \partial{Z}$

    $\partial{b} = \partial{Z}$

- Gradient Descent on m examples (cost function)

    $\frac{\partial{J(w,b)}}{\partial{w_1}} = \frac{1}{m}\sum_{i=1}^m \frac{\partial{L(a^{(i)}, y^{(i)})}}{\partial{w_1}}$


## Vectorization
- Do not use rank 1 array
  
  e.g.
  ```python
  a = np.random.randn(5) # rank 1 shape(5,)

  # use this bellow
  a = np.random.randn(5,1)
  a = np.random.randn(1,5)
  ```

- Broadcasting numpy
  a = np.random.randn(2,3) # shape = (2,3)
  b = np.random.randn(2,1) # shape = (2,1)
  c = a + b # shape = (2,3)

- Explanation of logistic cost function
  
    As proof in my notebook-logistic part