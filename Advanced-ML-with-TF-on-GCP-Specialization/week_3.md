# week 3

- Use the gcloud command to submit the training job either locally or to the cloud
  - local
    ```cmd
    gcloud ml-engine local train \
        --module-name=trainer.task \
        --package-path=/somedir/babyweight/trainer \
        -- \ # distinguish arguments for task.py file and ml-engine
        -- train_data_path etc.
        REST as before
    ```
  - cloud
    ```cmd
    gcloud ml-engine jobs submit training $JOBNAME \
        --region=$REGION \
        --module-name=trainer.task \
        --job-dir=$OUTPUR --staging-bucket=gs://$BUCKET \
        --scale-tier=BASIC \
        REST as before
    ```

- hyperparameter tuning
    ```cmd
    %bash
    OUTDIR=gs://$(BUCKET)/babyweight/hyperparam
    JOBNAME=babyweight_$(date -u +%y%m%d_%H%M%S)
    echo $OUTDIR $REGION $JOBNAME
    gsutil -m -rm -rf $OUTDIR
    gcloud ml-engine jobs submit training $JOBNAME \
      --region=$REGION \
      --module-name=trainer.task \
      --package-path=$(pwd)/babyweight/trainer
      --job-dir=$OUTDIR \
      --staging-bucket=gs://$BUCKET \
      --scale-tier=STANDARD_1 \
      --config=hyperparam.yaml \
      --runtime-version=$TFVERSION \
      -- \
      --bucket=${BUCKET} \
      --output_dir=${OUTDIR} \
      --rval_steps=10 \
      --train_examples=20000
    ```

- BigQuery ML

  Simplify model development with BigQuuery ML
  - Use familiar SQL for machine learning
  - Train models over all their data in BigQuery
  - Don't worry about hypertuning or feature transformations

  Supported features
  - StandardSQL and UDFs within the ML queries
  - Linear Regression(Forecasting)
  - Binary Logistic Regression(classification)
  - Model evaluation functions for standard metrics, including ROC and precision-recall curves
  - Model weight inspection
  - Festure distribution analysis through standard functions

- Deploying and Predicting with Cloud ML Engine
  - The serving_input_fn specifies what the caller of the predict() method must provide
  ```python
  def serving_input_fn():
    feature_placeholders = {
      'feature_1': tf.placeholder(tf.float32, [None]),
      'feature_2': tf.placeholder(tf.float32, [None])
    }
    features = {
      key: tf.expand_dims(tensor, -1)
      for key, tensor in feature_placeholder.items()
    }
    return tf.estimator.export.ServingInputReceiver(features,
          feature_placeholders)
  ```
  - Deploy a trained model to GCP
  ```shell
  MODEL_NAME="model_name"
  MODEL_VERSION="v1"
  MODEL_LOCATION="gs://${BUCKET}/path-to-model/export/exporter/.../"

  gclouud ml-engine models create ${MODEL_NAME} --regions $REGION

  gcloud ml-engine versions create ${MODEL_VERSION} --model ${MODEL_NAME} --origin ${MODEL_LOCATION}
  ```

  - Client code can make REST calls
  ```python
  credentials = GoogleCredentials.get_application_default()
  api = discovery.build('ml', 'v1', 
                        credentials=credentials,
                        discoveryServiceUrl = 'https://storage.googleapis.com/cloud-ml/discovery/ml-v1beta1_discovery.json')
  request_data = [
    'feature_1': value,
    'feature_2': value
  ]
  parent = 'project/%s/models/$s/versions/%s' % ('cloud-training-demos', 'model-name', 'version')
  response = api.projects().predict(body={'instances': request_data},name=parent).execute()
  ```