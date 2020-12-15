# week 1

- Setting up your Machine Learning Application

  - Bias and Variance

    Human level performance gets nearly 0% error, optimal(Bayes) error: 1% e.g.

- Regularizating your neural network

    - L2 regularization - weight decay

        $$
        \frac{\lambda}{2 m}\|\omega\|_{2}^{2}
        $$

    - L1 regularization

        $$
        \frac{\lambda}{2 m}\|w\|_{1}
        $$

    - Neural network

        Frobenius norm:
        $$
        \begin{aligned}
            &\frac{\lambda}{2 m} \sum_{l=1}^{L}\left\|w^{[l]}\right\|^{2}_F
            \\ \\ 
            & \left\|w^{[l]}\right\|^{2}_F = \sum_{i=1}^{n^{[l-1]}} \sum_{j=1}^{n^{[l]}}(w_{ij}^{[l]})^2
        \end{aligned}
        $$

        $$
        \begin{aligned}
            d w^{[l]} &= (\text{from back propagation}) + \frac{\lambda}{m}w^{[l]} \\
            w^{[l]} &:= w^{[l]} - \alpha d w^{[l]} \\
            w^{[l]} &:= w^{[l]} - \alpha[(\text{from back propagation})+\frac{\lambda}{m}w^{[l]}] \\
            &= w^{[l]} - \frac{\alpha \lambda}{m}w^{[l]} - \alpha(\text{from back propagation})\\
            &\text{we can see there, L2 norm is weight decaying}
        \end{aligned}
        $$

    - Dropout regularization

        - Implementing dropout ("Inverted dropout")

            $$
            \begin{aligned}
                d3 &= np.random.rand(a3.shape[0], a3.shape[1]) < keep-prob \\
                a3 &= np.multiply(a3, d3) \\
                a3 & /= keep-prob (inverted dropout)
            \end{aligned}
            $$
        - Making prediction at test time - no drop out
            $$
            \text{use} \: a3 \, /= keep-prob (inverted dropout) \,\text{to avoid}
            $$
    - other regularization methods
        - Data augmentation
        - Early stopping
        - Optimize cost function

- Setting up your optimization problem

    - normalizing inputs

        $$
        x :=x-\mu
        $$
        $$
        x /=\sigma^{2}
        $$

    - Vanishing / Exploding gradients

        $$
        \hat{y}=w^{[l]} \left[ \begin{array}{cc}{a} & {0} \\ {0} & {a}\end{array}\right]^{l-1} x
        $$
        If $a > 1, \hat{y} \, is \, exponential\, growing$ 

        If $a < 1,\hat{y} \, is \, exponential\, decresing$

    - Weight Initialization for Deep Network

        - Using Relu activation function
            $$
            Var{}\left(w_{i}\right)=\frac{2}{n}
            $$
            $$
            w^{[l]} = np.random.randn(shape) * np.sqrt(\frac{2}{n^{[l-1]}})
            $$

        - Using tangh activation function
            $$
            w^{[l]} = np.random.randn(shape) * np.sqrt(\frac{1}{n^{[l-1]}})
            $$

        - other
            $$ 
            w^{[l]} = np.random.randn(shape) * np.sqrt(\frac{2}{n^{[l]} + n^{[l]}})
            $$

    - Gradient check for a neural network

        $$
        \text { Take }W^{[1]}, b^{[1]}, \ldots, W^{[L]}, b^{[L]} \text { and reshape into a big vector } \theta.
        $$
        $$
        \text { Take } d W^{[1]}, d b^{[1]}, \ldots, d W^{[L]}, d b^{[L]} \text { and reshape into a big vector d } \theta
        $$

        Is $d\theta$ the slope of $J(\theta)$ ?

        For each i:
        $$
        \begin{aligned}
            d \theta_{approx}^{[i]} &= \frac{J(\theta_1, \theta_2,...,\theta_i+\epsilon, ...) - J(\theta_1, \theta_2,...,\theta_i-\epsilon, ...)}{2\epsilon} \\
            &\approx d \theta_{[i]} = \frac{\partial J}{\partial \theta_i}
        \end{aligned}
        $$
        take $\epsilon = 10^{-7}$ and check:
        $$
        \frac{\left\| d \theta_{approx} - d \theta \right\|_2}{\left\| d \theta_{approx} \right\|_2 + \left\| d \theta \right\|_2}
        $$
    
    - Gradient checking implementation notes

        $$
        \begin{aligned}
            &\text { - Don't use in training - only to debug } \\
            &\text { - If algorithm fails grad check, look at components to try to identify bug. } \\
            &\text { - Remember regularization. } \\
            &\text { - Doesn't work with dropout. } \\
            &\text { - Run at random initialization; perhaps again after some training. }
        \end{aligned}
        $$