# week 4

- Image generator in tensorflow
  
  directory -> labels
  ```python
  from tensorflow.keras.preprocessing.image
  import ImageDataGenerator

  train_datagen = ImageDataGenerator(rescale=1./255)
  train_generator = train_datagen.flow_from_directory(
      train_dir,
      target_size=(300, 300),
      batch_size=128,
      class_mode='binary'
  )

  test_datagen = ImageDataGenerator(rescale=1./255)
  validation_generator = test_datagen.flow_from_directory(
      validation_dir,
      target_size=(300, 300),
      batch_size=128,
      class_mode='binary'
  )
  ```

- Define a ConvNet to use complex images
  ```python
  model = tf.keras.models.Sequential()
    model.add(tf.keras.layer.Conv2D(64, (3,3), activation='relu', input_shape=[300, 300, 3]))
    model.add(tf.keras.layers.MaxPooling2D(2, 2))
    model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(2, 2))
    model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(2, 2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=512, activation='relu'))
    model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
  ```

- training (fit) model viafit_generator
  ```python
  history = model.git_generator(
      train_generator,
      step_per_epoch=8,
      epochs=15,
      validation_data=validation_generator,
      validation_steps=8,
      verbose=2
  )
  ``` 
- Using google colab
  ```python
  import numpy as np
  from google.colab import files
  from keras.preprocessing import image

  uploaded = files.upload()

  for fn in uploaded.keys():
    path = '/content/' + fn
    img = image.load_img(path, target_size=300, 300)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    classes = model.predict(images, batch_size=10)
    print(classes[0])
    if classes[0] > 0.5:
        print(fn + ' is a human.')
    else:
        print(fn + ' is a horse.')
  ```
  
- codes
  ```python
    import tensorflow as tf
    import os
    import zipfile

    DESIRED_ACCURACY = 0.999

    !wget --no-check-certificate \
        "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/happy-or-sad.zip" \
        -O "/tmp/happy-or-sad.zip"

    zip_ref = zipfile.ZipFile("/tmp/happy-or-sad.zip", 'r')
    zip_ref.extractall("/tmp/h-or-s")
    zip_ref.close()

    class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('acc')>DESIRED_ACCURACY):
        print("\nReached 99.9% accuracy so cancelling training!")
        self.model.stop_training = True

    callbacks = myCallback()

    model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    from tensorflow.keras.optimizers import RMSprop

    model.compile(loss='binary_crossentropy',
                    optimizer=RMSprop(lr=0.001),
                    metrics=['acc'])

    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    train_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
            "/tmp/h-or-s",  
            target_size=(150, 150), 
            batch_size=10,
            class_mode='binary')

    # Expected output: 'Found 80 images belonging to 2 classes'
    history = model.fit_generator(
      train_generator,
      steps_per_epoch=2,  
      epochs=15,
      verbose=1,
      callbacks=[callbacks])
  ```