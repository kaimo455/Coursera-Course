# week 2

- Simple computer vision - fashion mnist
  ```python
  
  import tensorflow as tf
  from tensorflow import keras

  fashion_mnist = keras.datasets.fashion_mnist
  (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
  train_images = train_images/255.
  test_images = test_images/255.
  ```

  ```python
  model = keras.Sequential()
  model.add(keras.layers.Flatten(input_shape=[28, 28]))
  model.add(keras.layer.Dense(units=128, activation=tf.nn.relu))
  model.add(kearas.layer.Dense(units=10, activation=tf.nn.softmax))
  
  model.compile(tf.train.AdamOptimizer(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
  model.fit(train_images, train_labels, epochs=5)
  model.evaluate(test_imagess, test_labels)
  ```

    Adding callback function to stop training when reaching some metrics
  ```python
  class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('loss') < 0.4):
            print('\nLoss is low so cancelling training!')
            self.model.stop_training = True

  callback = myCallback()
  model = keras.Sequential()
  model.add(keras.layers.Flatten(input_shape=[28, 28]))
  model.add(keras.layer.Dense(units=128, activation=tf.nn.relu))
  model.add(kearas.layer.Dense(units=10, activation=tf.nn.softmax))
  
  model.compile(tf.train.AdamOptimizer(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
  model.fit(train_images, train_labels, epochs=5, callbacks=[callback])
  model.evaluate(test_imagess, test_labels)
  ```
