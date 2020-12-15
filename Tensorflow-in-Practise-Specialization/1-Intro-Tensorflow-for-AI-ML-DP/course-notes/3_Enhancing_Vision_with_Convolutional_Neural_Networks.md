# week 3
- Simple example for CNN code
    ```python
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layer.Conv2D(64, (3,3), activation='relu', input_shape=[28, 28, 1]))
    model.add(tf.keras.layers.MaxPooling2D(2, 2))
    model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(2, 2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=128, activation='relu'))
    model.add(tf.keras.layers.Dense(units=10, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    model.summary()
    model.fit(train_images, train_labels, epochs=5)
    ```

- Visualizing the Convolutions and Pooling
  ```python
  import matplotlib.pyplot as plt
  f, axarr = plt.subplots(3, 4)
  FIRST_IMAGE = 0
  SECOND_IMAGE = 23
  THIRD_IMAGE = 28
  CONVOLUTION_NUMBER = 1
  from tensorflow.keras import models
  layer_outputs = [layer.output for layer in model.layers]
  activation_model = tf.keras.models.Model(inputs = model.input, outputs=layer_output)
  for x in range(0, 4):
    f1 = activation_model.predict(test_images[FIRST_IMAGE].reshape(1, 28, 28, 1))[x]
    axarr[0, x].imshow(f1[0, :, :, CONVOLUTION_NUMBER], cmap='inferno')
    axarr[0, x].grid(False)

    f2 = activation_model.predict(test_images[SECOND_IMAGE].reshape(1, 28, 28, 1))[x]
    axarr[1, x].imshow(f2[0, :, :, CONVOLUTION_NUMBER], cmap='inferno')
    axarr[1, x].grid(False)

    f3 = activation_model.predict(test_images[SECOND_IMAGE].reshape(1, 28, 28, 1))[x]
    axarr[2, x].imshow(f3[0, :, :, CONVOLUTION_NUMBER], cmap='inferno')
    axarr[2, x].grid(False)
  ```