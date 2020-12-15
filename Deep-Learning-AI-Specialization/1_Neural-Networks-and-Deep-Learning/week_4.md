# week 4

- Getting your matrix demensions right

    $$
    \begin{aligned}
        Z^{[l]}, A^{[l]} &: (n^{[l]}, m) \\ 
        d Z^{[l]}, d A^{[l]} &: (n^{[l]}, m) \\
        W^{[l]} &: (n^{[l]}, n^{[l-1]}) \\ 
        b^{[l]} &: (n^{[l]}, 1)
    \end{aligned}
    
    $$

- Why deep representations

    $$
    \begin{array}{l}{\text { Informally: There are functions you can compute with a }} \\ {\text { "small" L-layer deep neural network that shallower networks }} \\ {\text { require exponentially more hidden units to compute. }}\end{array}
    $$ 

- Building blocks of deep neural networks

    $$
    a^{[l-1]} \rightarrow (w^{[l]}, b^{[l]}) (cache: z^{[l]}) \rightarrow a^{[l]}
    $$

    $$
    d a^{[l]} \rightarrow (w^{[l]}, b^{[l]}, d z^{[l]}) (cache :z^{[l]}) \rightarrow d w^{[l]}, d b^{[l]} \rightarrow d a^{[l-1]} 
    $$

    ![Forward and backward functions](/images/forward-and-back-functions.png)

- Forward and Backward Propagation

    - Forward propagation for layer $l$

        input $a^{[l-1}]$

        output $a^{[l]}$, cache $(z^{[l]})$
        $$
        \begin{aligned}
            Z^{[l]} &= W^{[l]} \cdot A^{[l-1]} + b^{[l]} \\ 
            A^{[l]} &= g^{[l]}(Z^{[l]})
        \end{aligned}
        $$

    - Backward propagation for layer $l$

        input $d a^{[l]}$

        output $d a^{[l-1]}, d w^{[l]}, d b^{[l]}$

        $$
        \begin{aligned}
            d Z^{[l]} &= d A^{[l]} * g^{[l]\prime}(Z^{[l]}) \\ 
            d W^{[l]} &= \frac{1}{m} d Z^{[l]} A^{[l-1]T} \\ 
            d b^{[l]} &= \frac{1}{m}np.sum(d Z^{[l]}, axis=1, keeydims=True) \\
            d A^{[l-1]} &= W^{[l]T} d Z^{[l]}
        \end{aligned}
        $$

    - summary

        - Forward
        $$
        \begin{aligned} Z^{[1]} &=W^{[1]} X+b^{[1]} \\ A^{[1]} &=g^{[1]}\left(Z^{[1]}\right) \\ Z^{[2]} &=W^{[2]} A^{[1]}+b^{[2]} \\ A^{[2]} &=g^{[2]}\left(Z^{[2]}\right) \\ \bullet\\\bullet\\\bullet\\A^{[L]} &=g^{[L]}\left(Z^{[L]}\right)=\hat{Y} \end{aligned}
        $$

        - Backward
        $$
        \begin{aligned} d Z^{[L]} &=A^{[L]}-Y \\ d W^{[L]} &=\frac{1}{m} d Z^{[L]} A^{[L]^{T}} \\ d b^{[L]} &=\frac{1}{m} n p \cdot \operatorname{sum}\left(\mathrm{d} Z^{[L]}, a x i s=1, k e e p d i m s=\text {True}\right) \\ 
        d Z^{[L-1]}&=d W^{[L]^{T}} d Z^{[L]} g^{\prime[L]}\left(Z^{[L-1]}\right) \\\bullet\\\bullet\\\bullet\\
        d Z^{[1]} &=d W^{[2]^{T}} d Z^{[2]} g^{\prime[1]}\left(Z^{[1]}\right) \\ d W^{[1]} &=\frac{1}{m} d Z^{[1]} A^{[1]^{T}} \\ d b^{[1]} &=\frac{1}{m} n p \cdot \operatorname{sum}\left(\mathrm{d} Z^{[1]}, a x i s=1, k e e p d i m s=T r u e\right)
        \end{aligned}
        $$