- [Deep Learning for Sequences](#deep-learning-for-sequences)
  - [Introduction to RNN](#introduction-to-rnn)
    - [motivation for recurrent layers](#motivation-for-recurrent-layers)
    - [simple RNN and backpropagation](#simple-rnn-and-backpropagation)
  - [Modern RNNs](#modern-rnns)
    - [the training of RNNs is not that easy](#the-training-of-rnns-is-not-that-easy)
    - [dealing with vanishing and exploding gradients](#dealing-with-vanishing-and-exploding-gradients)
    - [modern RNNs: LSTM and GRU](#modern-rnns-lstm-and-gru)
  - [Applications of RNNs](#applications-of-rnns)
    - [practical use cases for RNNs](#practical-use-cases-for-rnns)

# Deep Learning for Sequences

## Introduction to RNN

### motivation for recurrent layers

- sequential data
  - text, video and audio
  - time series
- language model
- why not MLP?

  arbitrary length of sequence.

  we can use a window of fixed size as an input: heuristic and its not clear how to choose the width of the window.

- recurrent architecture

  ![](https://cdn.mathpix.com/snip/images/LSbcEfD5zobIPHKgfZPs4u6Lnl9sfHzSqeCrffdZCxQ.original.fullsize.png)

### simple RNN and backpropagation

- recurrent architecture

  ![](https://cdn.mathpix.com/snip/images/8HyfNRqVFJoxxoF8UlMyKsKJ6KLg6SqZfft6lD2ppc4.original.fullsize.png)

  ![](https://cdn.mathpix.com/snip/images/lOYZHcZ85BviDNOv2g8TbzCLq1gDRFDiDOGEqJ3mI90.original.fullsize.png)

  ![](https://cdn.mathpix.com/snip/images/BFpKH3NROiu_LcKvEsF8M1yTgHX5l6HEGP8umHfr334.original.fullsize.png)

- backpropagation through time (BPTT)

  ![](https://cdn.mathpix.com/snip/images/dkdx7f5XaWQc6kQvZdFqh0AZqMnhybXLUEN9qi6GTeA.original.fullsize.png)

  ![](https://cdn.mathpix.com/snip/images/EFrtG5sJBBixLqxpPwtra_BFJflvhD9n9AE-j_Zdgss.original.fullsize.png)

  ![](https://cdn.mathpix.com/snip/images/fKyLG3r7nw4aPT5lmsOZ06KvQejPA-Y1ukE4vDT-5fg.original.fullsize.png)

  $\sum_{k=0}^{t}\left(\prod_{i=k+1}^{t} \frac{\partial h_{i}}{\partial h_{i-1}}\right) \frac{\partial h_{k}}{\partial W}$

  https://cdn.mathpix.com/snip/images/d5bwf-pa-JNL_A_ncCE65NeQj0g3L0JyOEkjtEudq1c.original.fullsize.png

## Modern RNNs

### the training of RNNs is not that easy

- let's look at the gradient
  - gradient vanishing
  - exploding gradients

  $$
  \frac{\partial L_{t}}{\partial W} \propto \sum_{k=0}^{t}\left(\prod_{i=k+1}^{t} \frac{\partial h_{i}}{\partial h_{i-1}}\right) \frac{\partial h_{k}}{\partial W}
  $$

- summary
  - in practice, vanishing and exploding gradients are common for RNNs. These problems also occur in deep Feedforward NNs
  - vanishing gradients make the learning of long-range dependencies very difficult
  - exploding gradients make the leanring process very unstable and may even crash it

### dealing with vanishing and exploding gradients

- deal with exploding gradients
  - gradient clipping: threshold

    $$
    \begin{array}{l}{\text { If }\|g\|>\text { threshold: }} \\ {g \leftarrow \frac{\text { threshold }}{\|g\|} g}\end{array}
    $$

  - BPTT (computationally expensive)
  - truncated BPTT (chunks of the sequence instead of the whole sequence)

- gradient vanishing
  - LSTM, GRU
  - ReLU activation function
  - initialization of the recurrent weight matrix: initialize W with an orthogonal matrix
  - skip connections

### modern RNNs: LSTM and GRU

![](https://cdn.mathpix.com/snip/images/NTnlH0t3KxnHk320q5GV0YKTvhCixqlNE_l053fwB5M.original.fullsize.png)

- simple RNN

  ![](https://cdn.mathpix.com/snip/images/QGs70u9YiNpCBiMyh51DvuB5IXAzNJRC6gq1CuodRx8.original.fullsize.png)

- LSTM version 0

  ![](https://cdn.mathpix.com/snip/images/WQ4yySNSdrMVB-codZs1U4nj7RG58GVnfJ6X0NvpWvM.original.fullsize.png)

  ![](https://cdn.mathpix.com/snip/images/Gg-W2pjj2bcVZVS0ziyqCr7X7yXNz6UyoqIspdztvpQ.original.fullsize.png)

  ![](https://cdn.mathpix.com/snip/images/OgT-mKw-RFhssNgoStDxGNRFAWGpAbaaFfcglN88bMg.original.fullsize.png)

- LSTM forget sometimes

  ![](https://cdn.mathpix.com/snip/images/phiZV48cqVB9pw2qWbtupe6EAhIIzfKCMFmXEFLfOKQ.original.fullsize.png)

- LSTM extreme regimes

  ![](https://cdn.mathpix.com/snip/images/YLB0DMQ5T_QlxBie_uBp-4rMi2elIP8nbd_3nkxEhCw.original.fullsize.png)

  ![](https://cdn.mathpix.com/snip/images/MgLUQpG5B4dfD9vZxX5BEd5j_kIyZzQlojBEug7WZjY.original.fullsize.png)

- GRU

  ![](https://cdn.mathpix.com/snip/images/FTC2dPw58Pg6N78__lkJ8MXVxe2dzJJrmG0NCa94_Yc.original.fullsize.png)

  ![](https://cdn.mathpix.com/snip/images/VVHbzk_E9OxM3gnMHrV3dPXbkN2MlcVN4uS9Meoeiyo.original.fullsize.png)

- LSTM or GRU

  LSTM -> more flexible

  GRU -> less parameters

  first train LSTM second train GRU then compare them

- LSTM or GRU: stack more layers

  ![](https://cdn.mathpix.com/snip/images/5fj5w0F91tzxRYc_dcHtHDzDzSyw4uM8LtqU0wRzU0c.original.fullsize.png)

## Applications of RNNs

### practical use cases for RNNs

- elements-wise classification
  - input-sequence output-sequence
  - tasks: POS tagging, video frames classification

- sequence generation
  - input-xxx output-sequence
  - tasks:
    - character-based language model
    - word-nased language model
    - music generation
    - speech generation
    - handwriting generation
    - ...

- conditional sequence generation
  - input-some object output-sequence
  - tasks:
    - speech generation
    - handwriting generation
    - image captioning

- sequence classification
  - input-sequence output-label
  - tasks:
    - sentiment analysis
    - ...

- sequence translation
  - input-sequence output-sequence
  - tasks:
    - handwriting to text/text to handwriting
    - speech to text/text to speech
    - machine translation
  - input and output are not synchronized