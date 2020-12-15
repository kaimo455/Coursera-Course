import apache_beam as beam
import datetime, os

def to_csv(rowdict):
    import copy
    import hashlib
    no_ultrasound = copy.deepcopy(rowdict)
    w_ultrasound = copy.deepcopy(rowdict)

    CSV_COLUMNS = 'weight_pounds,is_male,mother_age,plurality,gestation_weeks'.split(',')

    no_ultrasound['is_male'] = 'Unknown'
    if rowdict['plurality'] > 1:
        no_ultrasound['plurality'] = 'Multiple(2+)'
    else:
        no_ultrasound['plurality'] = 'Single(1)'
    w_ultrasound['plurality'] = ['SIngle(1)', 'Twins(2)', 'Triplets(3)', 'Quadruplets(4)', 'Quintuplets(5)'][rowdict['plurality'] - 1]

    for result in [no_ultrasound, w_ultrasound]:
        data = ','.join([str(result[k]) if k in result else 'None' for k in CSV_COLUMNS])
        key = hashlib.sha224(data).hexdigest()
        yield str('{}, {}'.format(data, key))

def preprocess(in_test_mode):
    import shutil, os, subprocess
    job_name = 'preprocess-babyweight-features' + '-' + datetime.datetime.now().strftime('%y%m%d-%H%M%S')

    if in_test_mode:
        print('Launching local job ... hang on')
        OUTPUT_DIR = '.preproc'
        shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
        os.makedirs(OUTPUT_DIR)
    else:
        print('Launching Dataflow job {} ... hang on'.format(job_name))
        OUTPUT_DIR = 'gs://{0}/babyweight/preproc'.format(BUCKET)
        try:
            subprocess.check_call('gsutil -m rm -r {}'.format(OUTPUT_DIR).split())
        except:
            pass

    options = {
        'staging_location': os.path.join(OUTPUT_DIR, 'tmp', 'staging'),
        'temp_location': os.path.join(OUTPUT_DIR, 'tmp'),
        'job_name': job_name,
        'project': PROJECT,
        'teardown_policy': 'TEARDOWN_ALWAYS',
        'no_save_main_session': True
    }

    opts = beam.pipeline.PipelineOptions(flag=[], **options)
    if in_test_mode:
        RUNNER = 'DirectRunner'
    else:
        RUNNER = 'DataflowRunner'
    p = beam.Pipeline(RUNNER, options=opts)

    query = """
    SELECT
        weight_pounds,
        is_male,
        mother_age,
        plurality,
        gestation_weeks,
        ABS(FARM_FINGERPRINT(CONCAT(CAST(YEAR AS STRING), CAST(month AS STRING)))) AS hashmonth
    FROM
        publicadata.samples.natality
    WHERE year > 2000
    AND weight_pounds > 0
    AND mother_age > 0
    AND plurality > 0
    AND gestation_weeks > 0
    AND month > 0
    """

    if in_test_mode:
        query = query + 'LIMIT 100'
    
    for step in ['train', 'eval']:
        if step == 'train':
            selquery = 'SELECT * FROM ({}) WHERE MOD(ABS(hashmonth), 4) < 3'.format(query)
        else:
            selquery = 'SELECT * FROM ({}) WHERE MOD(ABS(hashmonth, 4) = 3)'.format(query)
        
        (p
         | '{}_read'.format(step) >> beam.io.Read(beam.io.BigQuerySource(query=selquery, use_standard_sql=True))
         | '{}_csv'.format(step) >> beam.FlatMap(to_csv)
         | '{}_out'.format(step) >> beam.io.Write(beam.io.WriteToText(os.path.join(OUTPUT_DIR, '{}.csv'.format(step))))
        )
    
    job = p.run()
    if in_test_mode:
        job.wait_until_finish()
        print('Done!')

preprocess(in_test_mode=True)


####################################
# Lab 5 Traning on Cloud ML Engine #
####################################

############
# setup.py #
############
from setuptools import setup
REQUIRED_PACKAGES = []
setup(
    name='babyweight',
    version='0.1',
    author='author',
    author_email='email',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='description',
    requires=[]
)
###########
# task.py #
###########
import argparse
import json
import os

import model

import tensorflow as tf
from tensorflow.contrib.learn.python.learn import learn_runner

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--bucket',
        help='GCS path to data. We assume that data is in gs://BUCKET/babyweight/preproc',
        required=True
    )
    parser.add_argument(
        '--output_dir',
        help='GCS location to write checkpoint and export models',
        required=True
    )
    parser.add_argument(
        '--train_examples',
        help='Number of examples (in thousands) to run the training job over. If this is more than actual # of examples available, it cycles through them. So specifying 1000 here when you have only 100k examples makes this 10 epochs.',
        type=int,
        default=5000
    )
    parser.add_argument(
        '--batch_size',
        help='Number of examples to computue gradient over.',
        type=int,
        default=512
    )
    parser.add_argument(
        '--nembeds',
        help='Embedding size of a cross of n key real-valued parameters',
        type=int,
        default=3
    )
    parser.add_argument(
        '--nnsize',
        help='Hidden layer sizes to use for DNN feature columns -- provide space-separated layers',
        nargs='+',
        type=int,
        default=[128, 32, 4]
    )
    parser.add_argument(
        '--pattern',
        help='Specify a pattern that has to be in input files. For example 00001-of will process only one shard',
        default='of'
    )
    parser.add_argument(
        '--job-dir',
        help='this model ignores this field, but it is required by gcloud',
        default='junk'
    )
    parser.add_argument(
        '--eval_steps',
        help='Positive number of steps for which to evaluate model. Default to None, which means to evaluate until input_fn raises an end-of-input exception',
        type=int,
        default=None
    )

    args = parser.parse_args()
    arguments = args.__dict__

    # unused args provided by service
    arguments.pop('job_dir', None)
    arguments.pop('job-dir', None)

    ## assign the arguments to the model variables
    output_dir = arguments.pop('output_dir')
    model.BUCKET = arguments.pop('bucket')
    model.BATCH_SIZE = arguments.pop('batch_size')
    model.TRAIN_STEPS = (arguments.pop('train_examples') * 1000)/model.BATCH_SIZE
    model.EVAL_STEPS = arguments.pop('eval_steps')
    print('Will train for {} steps using batch_size={}'.format(model.TRAIN_STEPS, model.BATCH_SIZE))
    model.PATTERN = arguments.pop('pattern')
    model.NEMBEDS = arguments.pop('nembeds')
    model.NNSIZE = arguments.pop('nnsize')
    print('Will use DNN size of {}'.format(model.NNSIZE))

    # Append trial_id to path if we are doing hptuning
    # This code can be removed if you are not using hyperparameter tuning
    output_dir = os.path.join(
        output_dir,
        json.loads(
            os.environ.get('TF_CONFIG', '{}')
        ).get('task', {}).get('trail', '')
    )

    # Run the training job
    model.train_and_evaluate(output_dir)


############
# model.py #
############
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import shutil
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

BUCKET = None
PATTERN = 'of'

# Determine CSV, label and key columns
CSV_COLUMNS = 'weight_pounds, is_male, mother_age, plurality, gestation_weeks'.split(',')
LABEL_COLUMN = 'weight_pounds'
KEY_COLUMN = 'key'

# Set default values for each CSV column
DEFAULTS = [[0,0], ['null'], [0,0], ['null'], [0,0], ['nokey']]

# Define some hyperparameters
TRAIN_STEPS = 10000
EVAL_STEPS = None
BATCH_SIZE = 512
NEMBEDS = 3
NNSIZE = [64, 16, 4]


# %writefile hyperparam.yaml
# trainingInput:
#     scaleTier: STANDARD_1
#     hyperparameters:
#         hyperparameterMetricTagL rmse
#         goal: MINIMIZE
#         maxTrials: 20
#         maxParallelTrials: 5
#         enableTrialEarlyStopping: True
#         params:
#         - parameterName: batch_size
#           type: INTEGER
#           minValue: 8
#           maxValue: 512
#           scaleType: UNIT_LOG_SCALE
#         - parameterName: NEMBEDS
#           type: INTEGER
#           minValue: 3
#           maxValue: 30
#           scaleType: UNIT_LINEAR_SCALE
#         - parameterName: nnsize
#           type: INTEGER
#           minValue: 64
#           maxValue: 512
#           scaleType: UNIT_LOG_SCALE


###################
# Deploying Model #
###################

%bash
MODEL_NAME = "babyweight"
MODEL_VERSION = "ml_on_gcp"
MODEL_LOCATION = $(gsutil ls gs://${BUCKET}/babyweight/trained_model/export/exporter/ | tail -1)
gcloud ml-engine models create ${MODEL_NAME} --regions $REGION
gcloud ml-engine versions create ${MODEL_VERSION} --model ${MODEL_NAME} --origin ${MODEL_LOCATION} --runtime-version $TFVERSION

# Use model to predict (online prediction)
from oauth2client.client import GoogleCredentials
imoprt requests
import json

MODEL_NAME = "babyweight"
MODEL_VERSION = "ml_on_gcp"

token = GoogleCredentials.get_application_default().get_access_token().access_token
api = 'https://ml.googleapis.com/v1/projects/{}/models/{}/versions/{}:predict'.format(PROJECT, MODEL_NAME, MODEL_VERSION)
header = {'Authorization': 'Bearer ' + token}
data = {
    'instances': [
        {
            'is_male': 'True',
            'monther_age': 26.0,
            'plurality': 2,
            'gestation_weeks': 39
        }
    ]
}