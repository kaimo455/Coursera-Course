- [Unsupervised representation learning](#unsupervised-representation-learning)
  - [Intro to unsupervised learning](#intro-to-unsupervised-learning)
    - [unsupervised learning: what it is and why bother](#unsupervised-learning-what-it-is-and-why-bother)
    - [autoencoders 101](#autoencoders-101)
  - [More autoencoders](#more-autoencoders)
    - [autoencoder applications](#autoencoder-applications)
    - [autoencoder applications: image generation, data visualization & more](#autoencoder-applications-image-generation-data-visualization--more)
  - [word embeddings](#word-embeddings)
    - [natural language processing primer](#natural-language-processing-primer)
    - [word embeddings](#word-embeddings-1)

# Unsupervised representation learning

## Intro to unsupervised learning

### unsupervised learning: what it is and why bother

- why bother?
  - find most relevant features
  - compress information
  - retrieve similar objects
  - generate new data samples
  - explore high-dimensional data

### autoencoders 101

- why do we ever need that?
  - compress data
  - dimensionality reduction

- linear case
  - e.g. matrix factorization, minimizing reconstruction error $\left\|X-U \cdot V^{T}\right\| \rightarrow \min _{U, V}$

- matrix decompositions: a different perspective

  ![](https://cdn.mathpix.com/snip/images/oHQdLTTeR71UV2mtlU0DnNZ1AkX5rraZOqT-Oo_2L6k.original.fullsize.png)

- deep autoencoder

  ![](https://cdn.mathpix.com/snip/images/DMHPwdtnmTXNbdeL0iYytDktvwVrDNGwATyKeaFw9uM.original.fullsize.png)

- Image2image: convolutional

  ![](https://cdn.mathpix.com/snip/images/n-_ZzHMDnwGhEnpVpYItHJaSBZcBMUWTpvPNwPejSOU.original.fullsize.png)

## More autoencoders

### autoencoder applications

- why do we ever need that?
  - learn some greate features
  - unsupervised pretraining

- expanding autoencoder

  ![](https://cdn.mathpix.com/snip/images/PaC9ug8sMVmt60mMMjLZxsX8YfEsikIwOwVjMjw3suE.original.fullsize.png)

  the naive approach will learn identity function, so we need to regularize

  - (sparse autoencoder)idea 1: L1 on activations, sparse code

    ![](https://cdn.mathpix.com/snip/images/F3J4739rDPL_3rpNm63eigvDdNighXkpd9Va1o95xWs.original.fullsize.png)

  - (redundant autoencoder)idea 2: noize/dropout, redundant code
 
    ![](https://cdn.mathpix.com/snip/images/XbSM9t-2PvkjzbUeAySICUlWu132JucGVHZTGSDn5xU.original.fullsize.png)

    $L=\|X-\operatorname{Dec}(\operatorname{Enc}(N \operatorname{oise}(X)))\|$

  - (denoizing autoencoder)idea 3: distort input, learn to fix distorsion

    ![](https://cdn.mathpix.com/snip/images/2OiIZ26gezx6l1fjMhVs87Do_FSUuCBXJ1WbD4ZO_2g.original.fullsize.png)

  - sparse vs. denoizing

    ![](https://cdn.mathpix.com/snip/images/HahmnNjE-0tcqr_DF57GG8QeToxkN-TZ0ARQzAh5Myo.original.fullsize.png)

- pretaining

  use autoencoder as initialization

  ![](https://cdn.mathpix.com/snip/images/qh8wj8GzoTZNvZb15p3PD6_kArqZFfTUG13Mx4GQVYE.original.fullsize.png)

  several encoder layers, starting from the first one

  ![](https://cdn.mathpix.com/snip/images/nsY19EvQdlk3glUhewDD4nlNVosSQEaBnr26B8vI9Fc.original.fullsize.png)

  - supervised pre-training (on similar task)
    - needs labels for similar problem
    - luckily, we have Imagenet and Model Zoo
      - Alas, it's only good for popular problems
  - unsupervised pretraining(autoencoder)
    - needs no labels at all
    - may learn features irrelevant to your problem
    - e.g. background sky color for object classification

### autoencoder applications: image generation, data visualization & more

- exploratory data analysis

  visualize data in hidden space

  ![](https://cdn.mathpix.com/snip/images/BqyUB3I1bEaIny3L-eSZghGD__j6TwePXCZhpTB5XCI.original.fullsize.png)

- image morphing with AE

  $$
  \begin{array}{l}{\text { Idea: }} \\ {\text { If Enc(image 1) = c1 }} \\ {\text { Enc(image2) = c1 }} \\ {\text { Than }} \\ {\text { maybe (c 1 + c2) /2 is a }} \\ {\text { semantic average of the }} \\ {\text { two images }}\end{array}
  $$

## word embeddings

### natural language processing primer

- text 101: tokens
  - bag of words

- text classification
  - regression
    - adult content filter(safe search)
    - detext age/gender/interests by search querries
    - convert movie review into "stars"
    - survey public opinion for the new iphone vs. old one

  - BoW + linear

    ![](https://cdn.mathpix.com/snip/images/tXRpuRwtmniF5JWVYpsQ7ZJwfkK8b33aIz-MiDpWgZc.original.fullsize.png)

- word embeddings

  we want a compact representation of text so that we could use if for neural networks.

  MDS, LLE, TSNE, etc.

### word embeddings

- sparse vector products
- word2vec

  ![](https://cdn.mathpix.com/snip/images/9f_kPa0JZ0VZZ6l_aa8SrhSZyCyu2Z30xtXQoTYy0mc.original.fullsize.png)

  ![](https://cdn.mathpix.com/snip/images/S5tn4lqqVrd8juaApo8Nd2a0uf-BzIoEc_ssE2F2tUI.original.fullsize.png)

- softmax problem

  ![](https://cdn.mathpix.com/snip/images/UTTyiNN6nyzjrYhL4GNnkCVuJhvyYg1Fgxo6J6Pusmo.original.fullsize.png)

- moew word embeddings
  - faster softmax
    - hierarchical softmax, negative samples
    - learn more
  - alternative models: GloVe
  - sentence level:
    - Doc2Vec, skip-thought(using rnn)