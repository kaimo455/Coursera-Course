- [Deep Learning for images](#deep-learning-for-images)
  - [motivation of convolutional layers](#motivation-of-convolutional-layers)
  - [our first CNN architechture](#our-first-cnn-architechture)
- [Modern CNNs](#modern-cnns)
  - [training tips and tricks for deep CNNs](#training-tips-and-tricks-for-deep-cnns)
  - [overview of modern CNN architectures](#overview-of-modern-cnn-architectures)
- [Applications of CNNs](#applications-of-cnns)
  - [learning new tasks with pre-trained CNNs](#learning-new-tasks-with-pre-trained-cnns)
  - [a glimpse of other computer vision tasks](#a-glimpse-of-other-computer-vision-tasks)

# Deep Learning for images

## motivation of convolutional layers

- image as a neural network input
- why not MLP -> convolutions will help

  convolusion is a dot product of a $kernel(filter)$

- convolutions have been used for a while: edge detection, sharpening, blurring
- covolution is similar to correlation
- convolution is translation equivariant
- convolutional layer in neural network
- backpropagation for CNN, sum up the gradients for the same shared weight
- convolutional vs. fully connected layer

## our first CNN architechture

- a color image input, $W$ is an image width, $H$ is an image height, $C_{in}$ is a number of input channels (e.g. 3 RBG channels)
- one kernel is not enough(parameters: $(W_k \times H_k \times C_{in} + 1) \times C_{out}$)
- one convolutional layer is not enough
- receptive field after N convolutional layers, if we stack $3 X 3$ kernels with stride-1, then the $N$th layer will be $2N+1 \times 2N+1$ receptive field. more general formula: $\text{receptive field}= K + (n-1) \times S \times (K-1)$
- we need to grow receptive field faster: stride
- how do we maintain translation invariance: pooling layer(not change depth/channels)
- backpropagation for max pooling layer

  maximum is not a differentiable function, so there is no gradient with respect to non maximum path neurons, since changing them slight does not affect the output.

- putting it all together into a simple CNN

  ![](https://cdn.mathpix.com/snip/images/omSN_O94YlG0rXl7obNHXBET-omFuibkJK7mAh3gtT8.original.fullsize.png)

  k5, p2, k5, p2, fc120, fc84, fc10

- learning deep representations

# Modern CNNs

## training tips and tricks for deep CNNs

- sigmoid activation: $\sigma(x)=\frac{1}{1+e^{-x}}$

  too large or too small value of $x$ leads to small gradient. $\frac{\partial \sigma}{\partial x}=\sigma(x)(1-\sigma(x))$

  - sigmoid neurons can saturate and lead to vanishing gradients
  - not zero-centered
  - $e^x$ is computationally expensive

- tanh activation: $\tanh (x)=\frac{2}{1+e^{-2 x}}-1$

  - zero-centered
  - but still pretty much like sigmoid

- ReLU activation: $f(x)=\max (0, x)$

  - fast to compute
  - gradients do not vanish for $x \gt 0$
  - provides faster convergence in practice
  - not zeros-centered
  - can die: if not activated, never updates

- Leaky ReLU activation: $f(x)=\max (a x, x)$

  - will not die
  - $a \neq 1$

- weights initializations

  - need to break summetry
  - linear models work best when inputs are normalized
  - neuron is a linear combination of inputs + activation
  - neuron output will be used by consecutive layers

  ![](https://cdn.mathpix.com/snip/images/jVow_kLRtBj4eBart3_4t_2sce1siWF-vZT0B52srDY.original.fullsize.png)

  ![](https://cdn.mathpix.com/snip/images/0CYPkNnhJNr0svE98eameQxHmpFpdB0G2KmxxOXhxic.original.fullsize.png)

  ![](https://cdn.mathpix.com/snip/images/CdA8hmAdaUq9AU9ReFJagwj40hkG9E7QFtx8SPrVwnQ.original.fullsize.png)

  ![](https://cdn.mathpix.com/snip/images/RTyuAm8fs91EHrIXMpR3DJq6xV-CDyQ7_vxjoXtbXXk.original.fullsize.png)

- batch normalization

  we know how to initialize our network to constrain variance, but what if it grows during backpropagation? batch normalization controls mean and variance of outputs before activations.

  let's normalize $h_i$ - neuron output before activation

  $$
  h_{i}=\gamma_{i}\left[\frac{h_{i}-\mu_{i}}{\sqrt{\sigma_{i}^{2}}}\right]+\beta_{i}
  $$

  we estimate $\mu_i$ and $\sigma_i^2$ from current training batch.

  during testing we will use an exponential moving average over train batches:

  $$
  0<\alpha<1 \quad \begin{array}{l}{\mu_{i}=\alpha \cdot \text { mean }_{\text {batch }}+(1-\alpha) \cdot \mu_{i}} \\ {\sigma_{i}^{2}=\alpha \cdot \text { variance }_{\text {batch }}+(1-\alpha) \cdot \sigma_{i}^{2}}\end{array}
  $$

- dropout
  - regularization technique to reduce overfitting
  - we keep neurons active with probability $p$
  - this way we sample the network during training and change only a subset of its parameters on every iteration
  
  ![](https://cdn.mathpix.com/snip/images/r6j_vY7v_9SaH5ue4OVyAzYwERXfq1oC0H5ff2f3Vok.original.fullsize.png)

- data augmentation

  NOTE: convolutional neural network are invariant to translation(cause pooling layer provides invariant to translation)

## overview of modern CNN architectures

- ImageNet classification dataset

- AlexNet(2012)
  - 11x11, 5x5, 3x3 convolutions
  - max pooling
  - dropout
  - data augmentation
  - ReLU activations
  - SGD with momentum

- VGG(2015)
  - similar to AlexNet, but only 3x3 convolutions, but los of filters
  - training silimar to AlexNet with additional multi-scale cropping

- Inception V3(2015)
  - use Inception block introduced in GoogLeNet(a.k.a. Inception V1)
  - Batch normalization
  - image distortions
  - RMSProp

  - how inception block works

    - capture interactions of input channels in one "pixel" of feature map
    - reduce the number of channels not hurting the quality of the model, because different channels can correlate
    - dimensionality reduction with added ReLU activation
    
      ![](https://cdn.mathpix.com/snip/images/xTwGUcyZIR9a2iSj1M6hqNo97onN00-Gp7SspMKvKqs.original.fullsize.png)

  - basic inception block

    all operations inside a block use stride 1 and enough padding to output the same spatial dimentsion ($W \times H$) of feature map. 4 different featrue maps are concatenated on depth at the end.

    ![](https://cdn.mathpix.com/snip/images/Kh20FsJ6ZQngzmMjwW0lP0p8Un3X27Ni-MgQ2cb1BkE.original.fullsize.png)

  - replace 5x5 convolutions

    5x5 convolutions are expensive, let's replace them with two layer of 3x3 convolutions which have an effective receptive field of 5x5.

    ![](https://cdn.mathpix.com/snip/images/sl8dZh6ReNbzvGTM5bH4qAXQE7Dnxz8eke-Ze1vT0HI.original.fullsize.png)

  - filter decomposition

    ![](https://cdn.mathpix.com/snip/images/VWOC_0B4a2ZvRIt0xFBshu9Kl2SZ-NT4sZwrE86nq2A.original.fullsize.png)

  - filter decomposition in Inception block

    3x3 convolutions are currently the most expensive parts, let's replace each 3x3 layer with 1x3 layer followed by 3x1 layer.

    ![](https://cdn.mathpix.com/snip/images/Qm0Kz0jlnDAKk1F46chBATcXFVbUwr4GDDFj9dEmGWo.original.fullsize.png)

- ResNet(2015)

  - introduces residual connections
  - 152 layers, few 7x7 convolutional layers, the rest are 3x3
  - batch normalization
  - max and average pooling

  - residual connections

    ![](https://cdn.mathpix.com/snip/images/u7eP3DtLIRmAiaBDTG0evpMKym8SIS1fJFIukIADQ0Y.original.fullsize.png)

# Applications of CNNs

## learning new tasks with pre-trained CNNs

- transfer learning
  - you need less data to train (fortr training only final MLP)
  - works if a domain of new task is similar to pre-trained one
  - that is we can partially reuse featrues extractor
  
    ![](https://cdn.mathpix.com/snip/images/x0_9wr7ov98wsuBONZkpdJM_9IKR2xqEnV1_b9LgiUQ.original.fullsize.png)

  - propagate all gradients with smaller learning rate
  - Keras has the weights of pre-trained VGG, Inception, ResNet architecture

- takeways

  ![](https://cdn.mathpix.com/snip/images/0V_6RC-OCZolezkNekdVuRRx1OMBcOu0fC5y36pnIxk.original.fullsize.png)

## a glimpse of other computer vision tasks

- other cimputer vision tasks
  - semantic segmentation
  - object classification + localization(bounding box)

- semantic segmentation

  ![](https://cdn.mathpix.com/snip/images/ZhidY4RjG_aUOszt5SeFT3fIvaE0JsPiF7oJCZHUftc.original.fullsize.png)

  let's add pooling, which acts like down-sampling, then we need unpooling/up-sampling

  ![](https://cdn.mathpix.com/snip/images/k1kSwF-ifNqtJFZ4I7jdSEjalmuxM-D6dcWjnITSn5g.original.fullsize.png)

  how to up-sampling:
  - fill with nearest neighbor values
  - max unpooling

    ![](https://cdn.mathpix.com/snip/images/bDZwcj39SA9PFY38YpbdlFNluH4VqE4yP4uhInZ3T1s.original.fullsize.png)

  - learnable unpooling
    - previous approaches are not data-driven
    - we can replace max pooling layer with convolutional layer that has a bigger stride

    ![](https://cdn.mathpix.com/snip/images/pCI6P6Qnx4K5VBj_d6S9GePd9V3qWmRVAgTZBfYGAtg.original.fullsize.png)

- objext classification + localization

  ![](https://cdn.mathpix.com/snip/images/c8VlPtCNa-fkcrXsiz1qG67BdFG01nrYvqwbCO_v9fY.original.fullsize.png)