# week 3

- Hyperparameters tuning

    - tuning process

        Try random values: Don't use a grid

    - appropriate scale for hyperparameters

        - picking hyperparameters at random
        
            e.g.
            $$
            \begin{aligned}
                r &= -4 \cdot np.random.rand() \in (-4 ,0)\\
                \alpha &= 10^{r}
            \end{aligned}
            $$

        - hyoperparameters for exponentially weighted average

            e.g.
            $$
            \begin{aligned}
                r \in [-1, -1] \\
                1 - \beta = 10^r \\
                \beta = 1 - 10^r
            \end{aligned}
            $$
    
    - Hyperparameters tuning in practive: Pandas vs. Caviar

        That is, babysitting one model once a time or training many models in parallel.

    
- Batch Normalization

    - Normalizing activations in a network

        Normalize $Z^{[l]}$

        Given some intermediate value in NN, $Z^{[1]}, Z^{[2]}, ..., Z^{[m]}$, in there we omit the layer notation $z^{[l]}$

        $$
        \begin{aligned}
            \mu &=\frac{1}{m} \sum_{i} Z^{(i)}\\
            \sigma^{2}&=\frac{1}{m} \sum_{i}\left(Z_{i}-\mu\right)^{2}\\
            Z^{(i)}_{norm} &= \frac{Z^{(i)}-\mu}{\sqrt{\sigma^{2}+\varepsilon}}
        \end{aligned}

        $$
        But you want to costomize variance and mean, so we introduce the $\gamma$ and $\beta$.
        $$

        \begin{aligned}
            \tilde{Z}^{(i)} &= \gamma Z^{(i)}_{norm} + \beta \\
            \gamma, \beta &\,\, learnable
        \end{aligned}
        $$

    - Fitting batch norm into a neural network

        - forward propagation
            $$
            \begin{aligned}
                X & \xrightarrow{w^{[l]}, b^{[l]}} Z^{[l]} \xrightarrow[\text{Batch Norm(BN)}]{\beta^{[l]}, \gamma^{[l]}} \tilde{Z}^{[l]} \rightarrow a^{[l]} = g^{[l]}(\tilde{Z}^{[l]})\\
                a^{[l]}&\xrightarrow{w^{[l+1]}, b^{[l+1]}} Z^{[l+1]} \xrightarrow[\text{Batch Norm(BN)}]{\beta^{[l+1]}, \gamma^{[l+1]}} \tilde{Z}^{[l+1]} \rightarrow a^{[l+1]} \rightarrow ...
            \end{aligned}
            $$

        - trainable parameters(not hyperparameters):

            $$
            w^{[1]}, b^{[1]}, w^{[2]}, b^{[2]},..., w^{[l]}, b^{[l]},\\
            \beta^{[1]}, \gamma^{[1]},\beta^{[2]}, \gamma^{[2]},...,\beta^{[l]}, \gamma^{[l]}
            $$

        - backward propagation
            $$
            \begin{aligned}
                d \beta^{[l]} &= \beta^{[l]} - \alpha d \beta^{[l]} \\
                d \gamma^{[l]} &= \gamma^{[l]} - \alpha d \gamma^{[l]}
            \end{aligned}
            $$

            In next formula derivation:

            $$
            f(y) \rightarrow y(\hat{x}, \gamma, \beta) \rightarrow \hat{x}\left(\mu, \sigma^{2}, x\right)
            $$

            The $y$ is $\tilde{Z}^{{[l]}}$, $\hat{x}$ is $Z^{[l]}_{norm}$, $x$ is $Z^{[l]}$

            ---

            part No.1 $y(\hat{x}, \gamma, \beta)$

            ---

            $$
            \begin{aligned} \frac{\partial f}{\partial \gamma} &=\frac{\partial f}{\partial y_{i}} \cdot \frac{\partial y_{i}}{\partial \gamma} \\ &=\sum_{i=1}^{m} \frac{\partial f}{\partial y_{i}} \cdot \hat{x}_{i}.\end{aligned}
            $$

            $$
            \begin{aligned} \frac{\partial f}{\partial \beta} &=\frac{\partial f}{\partial y_{i}} \cdot \frac{\partial y_{i}}{\partial \beta} \\ &=\sum_{i=1}^{m} \frac{\partial f}{\partial y_{i}} \end{aligned}
            $$

            $$
            \begin{aligned} \frac{\partial f}{\partial \hat{x}_{i}} &=\frac{\partial f}{\partial y_{i}} \cdot \frac{\partial y_{i}}{\partial \hat{x}_{i}} \\ &=\frac{\partial f}{\partial y_{i}} \cdot \gamma \end{aligned}
            $$

            ---

            part No.2 $\hat{x}\left(\mu, \sigma^{2}, x\right)$

            ---

            $$
            \frac{\partial f}{\partial \mu}=\frac{\partial f}{\partial \hat{x}_{i}} \cdot \frac{\partial \hat{x}_{i}}{\partial \mu}+\frac{\partial f}{\partial \sigma^{2}} \cdot \frac{\partial \sigma^{2}}{\partial \mu}
            $$

            $$
            \hat{x}_{i}=\frac{\left(x_{i}-\mu\right)}{\sqrt{\sigma^{2}+\epsilon}}
            $$

            $$
            \frac{\partial \hat{x}_{i}}{\partial \mu}=\frac{1}{\sqrt{\sigma^{2}+\epsilon}} \cdot(-1)
            $$

            $$
            \frac{\partial \sigma^{2}}{\partial \mu}=\frac{1}{m} \sum_{i=1}^{m} 2 \cdot\left(x_{i}-\mu\right) \cdot(-1) = 0
            $$

            $$
            \begin{aligned} \frac{\partial \hat{x}}{\partial \sigma^{2}} &=\sum_{i=1}^{m}\left(x_{i}-\mu\right) \cdot(-0.5) \cdot\left(\sigma^{2}+\epsilon\right)^{-0.5-1} \\ &=-0.5 \sum_{i=1}^{m}\left(x_{i}-\mu\right) \cdot\left(\sigma^{2}+\epsilon\right)^{-1.5} \end{aligned}
            $$

            $$
            \begin{aligned} \frac{\partial f}{\partial \mu} &=\left(\sum_{i=1}^{m} \frac{\partial f}{\partial \hat{x}_{i}} \cdot \frac{-1}{\sqrt{\sigma^{2}+\epsilon}}\right)+\left(\frac{\partial f}{\partial \sigma^{2}} \cdot \frac{1}{m} \sum_{i=1}^{m}-2\left(x_{i}-\mu\right)\right) \\ &=\sum_{i=1}^{m} \frac{\partial f}{\partial \hat{x}_{i}} \cdot \frac{-1}{\sqrt{\sigma^{2}+\epsilon}} \end{aligned}
            $$

            Then

            $$
            \frac{\partial f}{\partial x_{i}}=\frac{\partial f}{\partial \hat{x}_{i}} \cdot \frac{\partial \hat{x}_{i}}{\partial x_{i}}+\frac{\partial f}{\partial \mu} \cdot \frac{\partial \mu}{\partial x_{i}}+\frac{\partial f}{\partial \sigma^{2}} \cdot \frac{\partial \sigma^{2}}{\partial x_{i}}
            $$

            $$
            \frac{\partial \hat{x}_{i}}{\partial x_{i}}=\frac{1}{\sqrt{\sigma^{2}+\epsilon}}
            $$

            $$
            \frac{\partial \mu}{\partial x_{i}}=\frac{1}{m}
            $$

            $$
            \frac{\partial \sigma^{2}}{\partial x_{i}}=\frac{2\left(x_{i}-\mu\right)}{m}
            $$

            $$
            \frac{\partial f}{\partial x_{i}}=\left(\frac{\partial f}{\partial \hat{x}_{i}} \cdot \frac{1}{\sqrt{\sigma^{2}+\epsilon}}\right)+\left(\frac{\partial f}{\partial \mu} \cdot \frac{1}{m}\right)+\left(\frac{\partial f}{\partial \sigma^{2}} \cdot \frac{2\left(x_{i}-\mu\right)}{m}\right)
            $$

            ---

            Finally,

            $$
            \frac{\partial f}{\partial x_{i}}=\frac{\left(\sigma^{2}+\epsilon\right)^{-0.5}}{m}\left[m \frac{\partial f}{\partial \hat{x}_{i}}-\sum_{j=1}^{m} \frac{\partial f}{\partial \hat{x}_{j}}-\hat{x}_{i} \sum_{j=1}^{m} \frac{\partial f}{\partial \hat{x}_{j}} \cdot \hat{x}_{j}\right]
            $$

        - Why dose batch norm work

            - batch norm reduces the problem of input values changing - variance shift
            - batch norm as regularization

                - each mini-batch is scaled by the mean/variance computed on just that mini-batch
                - this add some noise to the values $z^{[l]}$ within that mini-batch. So similar to dropout, it adds some noise to each hidden layer's activations
                - this has a slight regularization effect

                bigger mini-batch size $\rightarrow$ reducing regularization effect

    - Batch norm at test time

        $\mu, \sigma$: estimate using expotentially weighted average(across mini-batch)

- Multi-class classification

    softmax activation function

    $$
    a^{[l]} = \frac{e^{z^{[l]}}}{\sum_{j=1}^{n^{[l]}}} e^{z^{[l]}}_j
    $$

    loss function for multi-calss classification with softmax:

    $$
    \mathcal{L}(\hat{y}, y) = - \sum_j^{\#class} y_j \log \hat{y}_j
    $$

- Deep Learning frameworks

    - Caffe/Caffe2
    - CNTK
    - DL4J
    - Keras
    - Lasagne
    - mxnet
    - PaddlePaddle
    - TensorFlow
    - Theano
    - Torch