- [The simplest neural network: MLP](#the-simplest-neural-network-mlp)
  - [the multilayer perception(MLP)](#the-multilayer-perceptionmlp)
  - [chain rule](#chain-rule)
  - [backpropogation](#backpropogation)
- [Matrix derivatives](#matrix-derivatives)
  - [efficient MLP implementation](#efficient-mlp-implementation)
  - [other matrix derivatives](#other-matrix-derivatives)
- [Tensorflow framework](#tensorflow-framework)
  - [what is TensorFlow](#what-is-tensorflow)
  - [Our first model in TensorFlow](#our-first-model-in-tensorflow)
- [Philosophy of deep learning](#philosophy-of-deep-learning)

# The simplest neural network: MLP

## the multilayer perception(MLP)

- let's recall linear binary classification

  ![](https://cdn.mathpix.com/snip/images/u_X2mfcTueVSaGQ9T05BYsENFtwwo91qVHxeSXh6tRk.original.fullsize.png)

- Logistic regression

  ![](https://cdn.mathpix.com/snip/images/cAyd_92VT9hWw7DWNchsO78kWpFYuj_GBG3iGvOLT18.original.fullsize.png)

- Triangle problem

  ![](https://cdn.mathpix.com/snip/images/OcmrDWfKOQ8VKq9pFmvmscyiW-ub-Q4QuPitkjQ0eA0.original.fullsize.png)

- these lines give us new features

  ![](https://cdn.mathpix.com/snip/images/IsCKqX8iQqvoGsGhsnUClH1FypIa3HGA9oa2g_4axEE.original.fullsize.png)

- we still don't know how to find there 4 lines

  ![](https://cdn.mathpix.com/snip/images/rDrID-I17jJ5Q1akmkH6vfPED2_havS1BOS3wd-pOdE.original.fullsize.png)

- our computation graph has a name

  ![](https://cdn.mathpix.com/snip/images/o7BpWejSsPio4Ln7aKSkPUBGlMomnTLchZC5ltd5a6Y.original.fullsize.png)

- why is it called a neuron?
- we need a non-linear activation function
- MLP overview

  ![](https://cdn.mathpix.com/snip/images/eVIr04NecN5fnr2WO-Sx0obUMiiyKwKIxH88ZdvNlwE.original.fullsize.png)

- how to train your MLP? -> SGD
  - other problems:
    - we can have many hidden layers (hyperparameters) -> we need to calculate gradients automatically
    - we can have mant neurons -> we need to calculate gradients fast

## chain rule

$$
\begin{array}{l}{\text { Let's take a composite function: }} \\ {\qquad \begin{aligned} z_{1} &=z_{1}\left(x_{1}, x_{2}\right) \\ z_{2} &=z_{2}\left(x_{1}, x_{2}\right) \\ p &=p\left(z_{1}, z_{2}\right) \\ & \frac{\partial p}{\partial x_{1}}=\frac{\partial p}{\partial z_{1}} \frac{\partial z_{1}}{\partial x_{1}}+\frac{\partial p}{\partial z_{2}} \frac{\partial z_{2}}{\partial x_{1}} \end{aligned}}\end{array}
$$

$$
\begin{array}{l}{\text { Example for } h(x)=f(x) g(x) :} \\ {\qquad \frac{\partial h}{\partial x}=\frac{\partial h}{\partial f} \frac{\partial f}{\partial x}+\frac{\partial h}{\partial g} \frac{\partial g}{\partial x}=g \frac{\partial f}{\partial x}+f \frac{\partial g}{\partial x}}\end{array}
$$

- derivatives computation graph

  ![](https://cdn.mathpix.com/snip/images/RBAHUmAYBDBu6SGAE98dpm-4bMyVFd9jq_ckjHzXf2Y.original.fullsize.png)

- let's go deeper

  ![](https://cdn.mathpix.com/snip/images/M08zeYVliBfWBofPHBiADUlCa4gGDudSEv57jgkaTNE.original.fullsize.png)

- derivatives computation graph

  ![](https://cdn.mathpix.com/snip/images/gAWp4SQqsoCENFEoUriEfO-ngcWCC-yhFujbzluTFJc.original.fullsize.png)

- summary
  - we can use chain rule to compute derivatives of composite functions
  - we can use a computation graph of derivatives to compute them automatically

## backpropogation

- we can reuse previous computations

  ![](https://cdn.mathpix.com/snip/images/D9FQ3emL73eHRe-eMU1VNeKfwNDsq7nsujlTM97Xovc.original.fullsize.png)

- this is called reverse-mode differentiation
  - in application to neural networks it has one more name: back-propogation
  - it works fast, because we reuse computations from previous steps
  - in fact, for each edge we compute its value only once, and multiply by its value exactly once.

- summary
  - back-propogation is a name for an automatic reverse-mode differentiation
  - back-propogation is fast
  - and this is how neural nets are trained today
  - in the next video we will look at MLP implementation details 

# Matrix derivatives

## efficient MLP implementation

- dense layer as a matrix multiplication
  - matrix multiplication can be done faster on both CPU(e.g. BLAS) and GPU(e.g. cuBLAS)
  - matrix multiplication with numpy is much faster than Python loops

- backward pass for a dense layer

  ![](https://cdn.mathpix.com/snip/images/nai9rF9T3FyUmmjLYRS9Pm73DOXaagiPFM9i5G_gTf0.original.fullsize.png)

  ![](https://cdn.mathpix.com/snip/images/2-TsCsB1geyN-6H1FDrxua0iaMZ12GrwS5UHohuiRks.original.fullsize.png)

- forawrd pass for mini-batches

  ![](https://cdn.mathpix.com/snip/images/X8Jm_PH9vite_54uESS5jIPM1ovKTmbWS9v3FvjNo5E.original.fullsize.png)

  $$
  \frac{\partial L_{b}}{\partial W}=X^{T} \frac{\partial L}{\partial Z} \quad X^{T}=\left(\begin{array}{ll}{x_{1,1}} & {x_{2,1}} \\ {x_{1,2}} & {x_{2,2}} \\ {x_{1,3}} & {x_{2,3}}\end{array}\right) \quad \frac{\partial L}{\partial Z}=\left(\begin{array}{cc}{\frac{\partial L}{\partial z_{1,1}}} & {\frac{\partial L}{\partial z_{1,2}}} \\ {\frac{\partial L}{\partial z_{2,1}}} & {\frac{\partial L}{\partial z_{2,2}}}\end{array}\right)
  $$

- backward pass for X (used in MLP)

  ![](https://cdn.mathpix.com/snip/images/JG1nPDqsKdvs-_0nQTig9qEl1CaeMHVToVsi1FMTQ5w.original.fullsize.png)

## other matrix derivatives

- Jacobian

  ![](https://cdn.mathpix.com/snip/images/8iCyHalfrWkkDGGdaYSml4Q2qUkxXZPGOiFP_k0xLx8.original.fullsize.png)

- metrix by matrix derivative

  ![](https://cdn.mathpix.com/snip/images/MbYYh2in_5m8cziTVAAndkO5Y9aZiCaf5FdtVJUMrWc.original.fullsize.png)

- chain rule for tensor derivative

  ![](https://cdn.mathpix.com/snip/images/6QJE7PgpVUTFyoecwWsOqhXe4pHk4_wy9xYOu_S72ng.original.fullsize.png)

- what's bad about it
  - our chain rule is a linear combination of matrices
  - crunching a lot of zeros
  
  ![](https://cdn.mathpix.com/snip/images/KBZS3e6cx-CNX1u97zkZTmBc9EQ5dfhvQDna3PuVpq4.original.fullsize.png)

- we're still fine
  - we need to calculate gradients of a scaler loss with respect to another scaler, vector, matrix or tensor.
  - this's how tf.gradients() in Tensorflow works.
  - deep learning frameworks have optimized versions of backward pass for standard layers.

# Tensorflow framework

## what is TensorFlow

- how the input looks like
  - placeholder

    This is placeholder for a tensor, which will be fed during graph execution. (e.g. input features)

    ```python
    x = tf.placeholder(tf.float32, (None, 10))
    ```

  - variable

    This is a tensor with some values that is updated during execution(e.g. weights matrix in MLP)

    ```python
    w = tf.get_variable('W', shape=(10, 20), dtype=tf.float32)
    w = tf.Variable(tf.random_uniform((10, 20), name='W'))
    ```

  - constant

    This is a tensor with a value, that cannot be changed

    ```python
    c = tf.constant(np.ones((4,4)))
    ```

- computational graph
  - TensorFLow creates a default graph after importing
    - all the operations will go there by default
    - use `tf.get_default_graph()` which returns an instance of tf.Graph
  - you can create your own graph variable and define operations there:

    ```python
    g = tf.Graph()
    with g.as_default():
        pass
    ```

  - you can clear the default graph use `tf.reset_default_graph()`

- operations and tensors

  every node in graph is an operation, listing all nodes with `tf.get_default_graph().get_operations()`, get the output tensor info use `tf.get_default_graph().get_operatons()[0].outputs`

- run a graph `tf.Session`

  ```python
  # create a session
  s = tf.InteractiveSession()
  # define a graph
  a = tf.constant(5.0)
  b = tf.constant(6.0)
  c = a * b
  # running a graph
  s.run(c)
  ```

  - operations are written in C++ and executed on CPU and GPU
  - tf.Session owns necessary resources to execute graph, that occupy RAM
  - it is important to release there resources when they are no longer required with tf.Session.close()

- initialization of variables

  - variables with initial value:
    - Tensor: `tf.Variable(tf.random_uniform((10, 20)), name='W')`
    - Initializer: `tf.get_variable('W', shape=(10, 20), dtype=tf.float32)`
  - you need to run some code to compute that initial value in graph execution environment. Use `session.run(tf.global_variables_initializer())`

## Our first model in TensorFlow

- optimizers in TensorFlow

  ```python
  tf.reset_default_graph()
  x = tf.get_variable('X', shape=(), dtype=tf.float32)
  f = x**2

  optimizer = tf.train.GradientDescentOptimizer(lr=0.1)
  step = optimizer.miniminze(f, var_list=[x])
  ```

- trainable variables

  ```python
  step = optimizer.minimize(f)
  # because all variables are trainable by default:
  x = tf.get_variable('X', shape=(), dtype=tf.float32, trainable=True)
  # get all trainable variables
  tf.trainable_variables()
  ```

- making gradient descent steps

  ```python
  s = tf.InteractiveSession()
  s.run(tf.global_variables_initializer())

  for i in range(10):
    _, curr_x, curr_f = s.run([step, x, f])
    print(curr_x, curr_f)
  ```

- logging with tf.Print

  ```python
  f = x**2
  f = tf.Print(f, [x, f], "x, f:")
  for i in range(10):
    s.run([step, f])
  ```

- logging with TensorBoard

  ```python
  tf.summary.scaler('curr_x', x)
  tf.summary.scaler('curr_f', f)
  summaries = tf.summary.merge_all()

  s = tf.InteractiveSession()
  summary_writer = tf.summary.FileWriter('logs/1', s.graph)
  s.run(tf.global_variables_initializer())

  for i in range(10):
    _, curr_summaries = s.run([step, summaries])
    summary_writer.add_summary(curr_summaries, i)
    summary_writer.flush()
  ```

- launching TensorBoard

  ```bash
  tensorboard --logdir=./logs
  ```

- solving a linear regression

  ![](https://cdn.mathpix.com/snip/images/gyZH8pNrH4OJNujG5wzY0ogAnYiZR3laHq_uVF8rEwg.original.fullsize.png)

  ![](https://cdn.mathpix.com/snip/images/lbtNLBcTtSmwtO9lkK-Uhx-83oNuaKx732muDHZD3Ro.original.fullsize.png)

- model checkpoints

  ![](https://cdn.mathpix.com/snip/images/ULg_BlzN-M708yTDLsW-0O5-xAsFiLttG4y9C_F49mQ.original.fullsize.png)

# Philosophy of deep learning

- what deep learning is and is not
  - book of grudges
    - no core theory
    - needs tons of data
    - computationally heavy
    - pathologicall overhyped
