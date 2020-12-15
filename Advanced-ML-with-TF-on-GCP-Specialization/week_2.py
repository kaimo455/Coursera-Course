import tensorflow as tf
import shutil
import numpy as np
import os

BUCKET = ''
PROJECT = ''
REGION = ''


os.environ['BUCKET'] = BUCKET
os.environ['PROJECT'] = PROJECT
os.environ[REGION] = REGION

# Determine CSV, label and key columns
CSV_COLUMNS = 'weight_pounds, is_male, mother_age, plurality, gestation_weeks'.split(',')
LABEL_COLUMN = 'weight_pounds'
KEY_COLUMN = 'key'

# Set default values for each CSV column
DEFAULTS = [[0,0], ['null'], [0,0], ['null'], [0,0], ['nokey']]
TRAIN_STEPS = 1000

# Create a input function reading a file using the Dataset API
# Then provide the results to the Estimator API
def read_dateset(filename, mode, batch_size=512):

    def _input_fn():
        
        def decode_csv(value_column):
            columns = tf.decode_csv(value_column, record_defaults=DEFAULTS)
            features = dict(zip(CSV_COLUMNS, columns))
            label = features.pop(LABEL_COLUMN)
            return features, label
        
        # create list of files that match pattern
        file_list = tf.gfile.Glob(filename)

        # Create dataset from file list
        dataset = (tf.data.TextLineDataset(file_list).map(decode_csv))

        if mode == tf.estimator.ModeKeys.TRAIN:
            num_epochs = None
            dataset = dataset.shuffle(buffer_size=10*batch_size)
        else:
            num_epochs = 1
        dataset = dataset.repeat(num_epochs).batch_size(batch_size)

        return dataset.make_one_shot_iterator().get_next()
    
    return _input_fn


# Define feature columns
def get_categorical(name, values):
    return tf.feature_column.indicator_column(
        tf.feature_column.categorical_column_with_vocabulary_list(name, values)
    )

def get_cols():
    return [
        get_categorical('is_male', ['True', 'False', 'Unknown']),
        tf.feature_column.numeric_column('month_age'),
        get_categorical('plurality', ['Single(1)', 'Twins(2)', 'Triplets(3)', 'Quadruplets(4)', 'Quintuplets(5)']),
        tf.feature_column.numeric_column('gestation_weeks')
    ]

# Features for DNNsLinearRegressor model
def get_wide_deep():
    is_male, mother_age, plurality, gestation_weeks = [
        tf.feature_column.categorical_column_with_vocabulary_list('is_male', ['True', 'False', 'Unknown']),
        tf.feature_column.numeric_column('mother_age'),
        tf.feature_column.categorical_column_with_vocabulary_list('plurality', ['Single(1)', 'Twins(2)', 'Triplets(3)', 'Quadruplets(4)', 'Quintuplets(5)']),
        tf.feature_column.numeric_column('gestation_weeks')
    ]
    # Discretize
    age_buckets = tf.feature_column.bucketized_column(mother_age, boundaries=np.arange(15,45,1).tolist())
    gestation_buckets = tf.feature_column.bucketized_column(gestation_weeks, boundaries=np.arange(17,41,1).tolisst())
    # Sparse columns are wides, have a linear relationship with the ouput
    wide = [
        is_male,
        plurality,
        age_buckets,
        gestation_buckets
    ]
    # Feature cross all the wide columns and embed into a lower dimension
    crossed = tf.feature_column.crossed_column(wide, hash_bucket_size=20000)
    embed = tf.feature_column.embedding_column(crossed, 3)

    # Continuous columns are deep, have a complex relationship with the output
    deep = [
        mother_age,
        gestation_weeks,
        embed
    ]
    
    return wide, deep

def serving_input_fn():
    feature_placeholders = {
        'is_male': tf.placeholder(tf.string, [None]),
        'mother_age': tf.placeholder(tf.float32, [None]),
        'plurality': tf.placeholder(tf.string, [None]),
        'gestation_weeks': tf.placeholder(tf.float32, [None])
    }
    features = {
        key: tf.expand_dims(tensor, -1) for key, tensor in feature_placeholders.items()
    }
    return tf.estimator.export.ServingInputReceiver(features, feature_placeholders)

# train and evaluate for DNNRegressor
def train_and_evaluate(output_dir):
    EVAL_INTERVAL = 300
    run_config = tf.estimator.RunConfig(save_checkpoints_secs = EVAL_INTERVAL, keep_checkpoint_max=3)
    estimator = tf.estimator.DNNRegressor(
        model_dir=output_dir,
        feature_columns = get_cols(),
        hidden_units=[64, 32],
        config=run_config
    )
    train_spec = tf.estimator.TrainSpec(
        input_fn=read_dateset('train.csv', mode=tf.estimator.ModeKeys.TRAIN),
        max_steps=TRAIN_STEPS
    )
    exporter = tf.estimator.LatestExporter('exporter', serving_input_fn)
    eval_spec = tf.estimator.EvalSpec(
        input_fn=read_dateset('eval.csv', mode=tf.estimator.ModeKeys.EVAL),
        steps=None,
        start_delay_secs=60,
        throttle_secs=EVAL_INTERVAL,
        exporters=exporter
    )
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

# train and evaluate for DNNLinearCombinedRegressor
def train_and_evaluate(output_dir):
    wide, deep = get_wide_deep()
    EVAL_INTERVAL = 300
    run_config = tf.estimator.RunConfig(save_checkpoints_secs=EVAL_INTERVAL, keep_checkpoint_max=3)
    estimator = tf.estimator.DNNLinearCombinedRegressor(
        model_dir=output_dir,
        linear_feature_columns=wide,
        dnn_feature_columns=deep,
        config=run_config
    )
    train_spec = tf.estimator.TrainSpec(
        input_fn=read_dateset('train.csv', mode=tf.estimator.ModeKeys.TRAIN),
        max_steps=TRAIN_STEPS
    )
    exporter = tf.estimator.LatestExporter('exporter', serving_input_fn)
    eval_spec = tf.estimator.EvalSpec(
        input_fn=read_dateset('eval.csv', mode=tf.estimator.ModeKeys.EVAL),
        steps=None,
        start_delay_secs=60,
        throttle_secs=EVAL_INTERVAL,
        exporter=exporter
    )
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

shutil.rmtree('babyweight_trained', ignore_errors=True)
train_and_evaluate('babyweight_trained')

from google.datalab.ml import TensorBoard
TensorBoard().start('./babyweight_trained')
    