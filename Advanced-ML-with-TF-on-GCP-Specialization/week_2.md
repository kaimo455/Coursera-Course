# week 2

-  creating dataset
- develop TensorFlow code on a small subset of data, then scale it out to the cloud. Example:(using RAND() to threshold number of samples)
    ```sql
    SELECT
        *
    FROM
        `bigquery-sample.airline_ontime_data.flight`
    WHERE
        MOD(ABS(FARM_FINGERPRINT(date)), 10) < 8
    AND
        RAND() < 0.01
    ```

- TensorFlow Toolkit hierarchy:
    - tf.estimator
    - tf.layers, tf.losses, tf.metrics
    - core TensorFlow (Python)
    - Core TensorFlow (C++)

- Example of an estimator API ML model
    ```python
    import tensorflow as tf

    # Define input feature columns
    featcols = [tf.feature_column.numeric_column('sq_footage')]

    # Instantiate Linear Regression Model
    model = tf.estimator.LinearRegressor(featcols, './model_trained')

    # Train
    def train_input_fn():
        # do something
        return features, labels
    model.train(train_input_fn, steps=100)

    # Predict
    def pred_input_fn():
        # do something
        return features
    out = model.predict(pred_input_fn)

    # As can be seen above, the reason of using the function to load data is avoid memery limitation when holding massive data in memery.
    ```

    - If you know the complete vocabulary beforehand:    
        ```python
        tf.feature_column.categorical_column_with_vocabulary_list('zipcode', vocabulary_list=['83452', '72345', '87654', '23451'])
        ```
    
    - If your data is already indexed, i.e., has integers in [0-N):
        ```python
        tf.feature_column.categorical_column_with_identity('stateId', num_buckets=50)
        ```

    - To pass in a categorical column into a DNN, one option is to one-hot encode it:
        ```python
        tf.feature_column.indicator_column(my_categorical_column)
        ```

- Read CSV file
    ```python
    CSV_COLUMNS = ['sqfootage', 'city', 'amount']
    LABEL_COLUMN = 'amount'
    DEFAULTS = [[0.0], ['na'], [0.0]]

    def read_dataset(filename, mode, batch_size=512):
        def decode_csv(value_column):
            columns = tf.decode_csv(value_column, record_defaults=DEFAULTS)
            features = dict(zip(CSV_COLUMNS, columns))
            label = features.pop(LABEL_COLUMN)
            return features, label

        dataset = tf.data.TextLineDataset(filename).map(decode_csv)

        if mode == tf.estimator.Modekeys.TRAIN:
            num_epochs = None
            dataset = dataset.shuffle(buffer_size=10*batch_size)
        else:
            num_epochs = 1
        dataset = dataset.repeat(num_epochs).batch(batch_size)
        
        return dataset.make_one_shot_iterator().get_next()
    ```

- Training and evaluating
    - Distribute the graph
    - Share variables
    - Evaluate occasionally
    - Handle machine failures
    - Create checkpoint files
    - Recover from failures
    - Save summaries for TensorBoard
    ```python
    estimator = tf.estimator.LinearRegressor(
        model_dir=output_dir,
        feature_columns=feature_cols
    )

    train_spec = tf.estimator.TrainSpec(
        input_fn=read_dataset('gs://.../train*',       
            mode=tf.contrib.learn.ModeKeys.TRAIN),
        max_steps=num_train_steps
    )

    exporter = ...
    eval_spec = tf.estimator.EvalSpec(
        input_fn=read_dataset('gs://.../valid*',
            mode=tf.contrib.learn.ModeKeys.EVAL),
        steps=None,
        start_delay_secs=60,
        throttle_secs=600,
        exporters=exporter
    )
    tf.estimator.train_and_evaluate(
        estimator,
        train_spec,
        eval_spec
    )
    ```
- Two types of features: Dense and sparse
- DNNs good for dense, highly-correlated inputs
- Linear models are better at handling sparse, independent features
- Wide and Deep model - DNNLinearCombinedClassifier
  ```python
    model = tf.estimator.DNNLinearCombinedClassifier(
        model_dir = ...,
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=[100, 50]
    )
  ```