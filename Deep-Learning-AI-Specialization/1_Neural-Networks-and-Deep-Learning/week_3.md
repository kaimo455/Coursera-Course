# week 3

## Shallow Neural Network
- Notation
    $m$ is the number of samples

    $X = [x^{[1]},x^{[2]},x^{[...]},x^{[m]}]$ stack in columns

    $x^{[i]} \in \reals^{n*1}$

    $X \in \reals^{n*m}$

    $W \in \reals^{units\_in\_this\_layer*units\_in\_last\_layer}$

    $Z = W \cdot X \in \reals^{units\_in\_this\_layer * m}$

    A representation of each leayers, forward propagation:

    $$
    \begin{aligned} Z^{[1]} &=W^{[1]} X+b^{[1]} \\ A^{[1]} &=\sigma\left(Z^{[1]}\right) \\ Z^{[2]} &=W^{[2]} A^{[1]}+b^{[2]} \\ A^{[2]} &=\sigma\left(Z^{[2]}\right) \end{aligned}
    $$

- Activation functions
  
    - sigmoid - for binary ouput
        $$
        a=\operatorname{sigmoid}(z)=\frac{1}{1+e^{-2}}
        $$

    - tanh
        $$
        a=\tanh (z)=\frac{e^{z}-e^{-z}}{e^{z}+e^{-z}}
        $$
    
    - relu
        $$
        a=\operatorname{max}(0, z)
        $$

    - leaky relu
        $$
        a=\max \left(0.01z, z\right)
        $$

- Derivatives of activation functions

    - sigmoid
        $$
        {g}(z)=\frac{1}{1+e^{-z}}
        $$
        $$
        \frac{d}{d z} g(z){=\frac{1}{1+e^{-z}}\left(1-\frac{1}{1+e^{-z}}\right)}{=g(z)(1-g(z))}
        $$

    - tanh
        $$
        \frac{d}{d z} g(z)=1-\tanh (z) )^{2}{=1-g(z)^2}
        $$

    - relu
        $$
        \frac{d}{d z} g(z)=g^{\prime}(z)=\left\{\begin{array}{ll}{0} & {\text { if } z<0} \\ {1} & {\text { if } z\geqslant0}\end{array}\right.
        $$

    - leaky relu
        $$
        \frac{d}{d z} g(z)=g^{\prime}(z)=\left\{\begin{array}{cl}{0.01} & {\text { if } z<0} \\ {1} & {\text { if } z \geqslant 0}\end{array}\right.
        $$
- Gradient descent for neural network
  
    - forward propagation

        $$
        \begin{aligned} Z^{[1]} &=W^{[1]} X+b^{[1]} \\ A^{[1]} &=\sigma\left(Z^{[1]}\right) \\ Z^{[2]} &=W^{[2]} A^{[1]}+b^{[2]} \\ A^{[2]} &=\sigma\left(Z^{[2]}\right) \end{aligned}
        $$

    - back propagation - logistic regression

        $$
        \begin{aligned}d Z^{[2]} &=A^{[2]}-Y \\ d W^{[2]} &=\frac{1}{m} d Z^{[2]} A^{[1] T} \\ d b^{[2]} &= \frac{1}{m} np.sum(d Z^{[2]}, axis=1, keepdims=True)  \\ d Z^{[1]} &= W^{[2] T} d Z^{[2]} * g^{[1] \prime}(Z^{[1]}) \\ d W^{[1]} &= \frac{1}{m} d Z^{[1]}X^{T} \\ d b^{[1]} &= \frac{1}{m}np.sum(d Z^{[1]}, axis=1, keepdims=True) \end{aligned}
        $$

- Backpropagation intuition

    - cost function to $a$
    $$
    \begin{aligned}
        \frac{d}{d a} \mathcal{L}(a, y) &=-y \log a-(1-y) \log (1-a) \\ &=-\frac{y}{a}+\frac{1-y}{1-a}
    \end{aligned}
    $$

    - $a$ to $z$,  $z = w^T x+b$
    $$
    \begin{aligned}
        d z &=d a \cdot g^{\prime}(z) \\
        \frac{d \mathcal{L}}{d z} &= \frac{d \mathcal{L}}{d a} \cdot \frac{d a}{d z}
    \end{aligned}
    $$

    - $d w$ and $d b$

    $$
    \begin{array}{l}{d w=d z \cdot x} \\ {d b=d z}\end{array}
    $$

    - summary

    $$
    \begin{array}{l}{d z^{[2]}=a^{[2]}-y} \\ {d W^{[2]}=d z^{[2]} a^{[1]^{T}}} \\ {d b^{[2]}=d z^{[2]}} \\ {d z^{[1]}=W^{[2] T} d z^{[2]} * g^{[1] \prime}\left(z^{[1]}\right)} \\ {d W^{[1]}=d z^{[1]} x^{T}} \\ {d b^{[1]}=d z^{[1]}} \end{array}
    $$

- random initialization

    - what happends if you initialize weights to zero?

        all to zero, then $a^{[1]}_1 = a^{[1]}_2$, then $d Z^{[1]}_1 = d Z^{[1]}_2$, that is symmetric.

    