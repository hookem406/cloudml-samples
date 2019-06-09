
#%%
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#%% [markdown]
# # Getting started: Training and prediction with Keras in AI Platform
# 
# <img src="https://storage.googleapis.com/cloud-samples-data/ml-engine/census/keras-tensorflow-cmle.png" alt="Keras, TensorFlow, and AI Platform logos" width="300px">
# 

# <table align="left">
#   <td>
#     <a href="https://cloud.google.com/ml-engine/docs/tensorflow/getting-started-keras">
#       <img src="https://cloud.google.com/_static/images/cloud/icons/favicons/onecloud/super_cloud.png"
#            alt="Google Cloud logo" width="32px"> Read on cloud.google.com
#     </a>
#   </td>
#   <td>
#     <a href="https://colab.research.google.com/github/GoogleCloudPlatform/cloudml-samples/blob/master/notebooks/tensorflow/getting-started-keras.ipynb">
#       <img src="https://cloud.google.com/ml-engine/images/colab-logo-32px.png" alt="Colab logo"> Run in Colab
#     </a>
#   </td>
#   <td>
#     <a href="https://github.com/GoogleCloudPlatform/cloudml-samples/blob/master/notebooks/tensorflow/getting-started-keras.ipynb">
#       <img src="https://cloud.google.com/ml-engine/images/github-logo-32px.png" alt="GitHub logo">
#       View on GitHub
#     </a>
#   </td>
# </table>
#%% [markdown]
# ## Overview
# 
# This tutorial shows how to train a neural network on AI Platform
# using the Keras sequential API and how to serve predictions from that
# model.
# 
# Keras is a high-level API for building and training deep learning models.
# [tf.keras](https://www.tensorflow.org/guide/keras) is TensorFlow’s
# implementation of this API.
# 
# The first two parts of the tutorial walk through training a model on Cloud
# AI Platform using prewritten Keras code, deploying the trained model to
# AI Platform, and serving online predictions from the deployed model.
# 
# The last part of the tutorial digs into the training code used for this model and ensuring it's compatible with AI Platform. To learn more about building
# machine learning models in Keras more generally, read [TensorFlow's Keras
# tutorials](https://www.tensorflow.org/tutorials/keras).
#%% [markdown]
# ### Dataset
# 
# This tutorial uses the [United States Census Income
# Dataset](https://archive.ics.uci.edu/ml/datasets/census+income) provided by the
# [UC Irvine Machine Learning
# Repository](https://archive.ics.uci.edu/ml/index.php). This dataset contains
# information about people from a 1994 Census database, including age, education,
# marital status, occupation, and whether they make more than $50,000 a year.
#%% [markdown]
# ### Objective
# 
# The goal is to train a deep neural network (DNN) using Keras that predicts
# whether a person makes more than $50,000 a year (target label) based on other
# Census information about the person (features).
# 
# This tutorial focuses more on using this model with AI Platform than on
# the design of the model itself. However, it's always important to think about
# potential problems and unintended consequences when building machine learning
# systems. See the [Machine Learning Crash Course exercise about
# fairness](https://developers.google.com/machine-learning/crash-course/fairness/programming-exercise)
# to learn about sources of bias in the Census dataset, as well as machine
# learning fairness more generally.
#%% [markdown]
# ### Costs
# 
# This tutorial uses billable components of Google Cloud Platform (GCP):
# 
# * AI Platform
# * Cloud Storage
# 
# Learn about [AI Platform
# pricing](https://cloud.google.com/ml-engine/docs/pricing) and [Cloud Storage
# pricing](https://cloud.google.com/storage/pricing), and use the [Pricing
# Calculator](https://cloud.google.com/products/calculator/)
# to generate a cost estimate based on your projected usage.
#%% [markdown]
# ## Before you begin
# 
# You must do several things before you can train and deploy a model in
# AI Platform:
# 
# * Set up your local development environment.
# * Set up a GCP project with billing and the necessary
#   APIs enabled.
# * Authenticate your GCP account in this notebook.
# * Create a Cloud Storage bucket to store your training package and your
#   trained model.
# 
#%% [markdown]
# ### Set up your local development environment
# 
# **If you are using Colab or AI Platform Notebooks**, your environment already meets
# all the requirements to run this notebook. You can skip this step.
#%% [markdown]
# **Otherwise**, make sure your environment meets this notebook's requirements.
# You need the following:
# 
# * The Google Cloud SDK
# * Git
# * Python 3
# * virtualenv
# * Jupyter notebook running in a virtual environment with Python 3
# 
# The Google Cloud guide to [Setting up a Python development
# environment](https://cloud.google.com/python/setup) and the [Jupyter
# installation guide](https://jupyter.org/install) provide detailed instructions
# for meeting these requirements. The following steps provide a condensed set of
# instructions:
# 
# 1. [Install and initialize the Cloud SDK.](https://cloud.google.com/sdk/docs/)
# 
# 2. [Install Python 3.](https://cloud.google.com/python/setup#installing_python)
# 
# 3. [Install
#    virtualenv](https://cloud.google.com/python/setup#installing_and_using_virtualenv)
#    and create a virtual environment that uses Python 3.
# 
# 4. Activate that environment and run `pip install jupyter` in a shell to install
#    Jupyter.
# 
# 5. Run `jupyter notebook` in a shell to launch Jupyter.
# 
# 6. Open this notebook in the Jupyter Notebook Dashboard.
#%% [markdown]
# ### Set up your GCP project
# 
# **The following steps are required, regardless of your notebook environment.**
# 
# 1. [Select or create a GCP project.](https://console.cloud.google.com/cloud-resource-manager)
# 
# 2. [Make sure that billing is enabled for your project.](https://cloud.google.com/billing/docs/how-to/modify-project)
# 
# 3. [Enable the AI Platform ("Cloud Machine Learning Engine") and Compute Engine APIs.](https://console.cloud.google.com/flows/enableapi?apiid=ml.googleapis.com,compute_component)
# 
# 4. Enter your project ID in the cell below. Then run the  cell to make sure the
# Cloud SDK uses the right project for all the commands in this notebook.
# 
# **Note**: Jupyter runs lines prefixed with `!` as shell commands, and it interpolates Python variables prefixed with `$` into these commands.

#%%
PROJECT_ID = "wgt-model" #@param {type:"string"}
get_ipython().system(' gcloud config set project $PROJECT_ID')

#%% [markdown]
# ### Authenticate your GCP account
# 
# **If you are using AI Platform Notebooks**, your environment is already
# authenticated. Skip this step.
#%% [markdown]
# **If you are using Colab**, run the cell below and follow the instructions
# when prompted to authenticate your account via oAuth.
# 
# **Otherwise**, follow these steps:
# 
# 1. In the GCP Console, go to the [**Create service account key**
#    page](https://console.cloud.google.com/apis/credentials/serviceaccountkey).
# 
# 2. From the **Service account** drop-down list, select **New service account**.
# 
# 3. In the **Service account name** field, enter a name.
# 
# 4. From the **Role** drop-down list, select
#    **Machine Learning Engine > AI Platform Admin** and
#    **Storage > Storage Object Admin**.
# 
# 5. Click *Create*. A JSON file that contains your key downloads to your
# local environment.
# 
# 6. Enter the path to your service account key as the
# `GOOGLE_APPLICATION_CREDENTIALS` variable in the cell below and run the cell.

#%%
import sys

# If you are running this notebook in Colab, run this cell and follow the
# instructions to authenticate your GCP account. This provides access to your
# Cloud Storage bucket and lets you submit training jobs and prediction
# requests.

if 'google.colab' in sys.modules:
  from google.colab import auth as google_auth
  google_auth.authenticate_user()

# If you are running this notebook locally, replace the string below with the
# path to your service account key and run this cell to authenticate your GCP
# account.
else:
  get_ipython().run_line_magic('env', "GOOGLE_APPLICATION_CREDENTIALS ''")

#%% [markdown]
# ### Create a Cloud Storage bucket
# 
# **The following steps are required, regardless of your notebook environment.**
# 
# When you submit a training job using the Cloud SDK, you upload a Python package
# containing your training code to a Cloud Storage bucket. AI Platform runs
# the code from this package. In this tutorial, AI Platform also saves the
# trained model that results from your job in the same bucket. You can then
# create an AI Platform model version based on this output in order to serve
# online predictions.
# 
# Set the name of your Cloud Storage bucket below. It must be unique across all
# Cloud Storage buckets. 
# 
# You may also change the `REGION` variable, which is used for operations
# throughout the rest of this notebook. Make sure to [choose a region where Cloud
# AI Platform services are
# available](https://cloud.google.com/ml-engine/docs/tensorflow/regions). You may
# not use a Multi-Regional Storage bucket for training with AI Platform.

#%%
BUCKET_NAME = "<your-bucket-name>" #@param {type:"string"}
REGION = "us-central1" #@param {type:"string"}

#%% [markdown]
# **Only if your bucket doesn't already exist**: Run the following cell to create your Cloud Storage bucket.

#%%
get_ipython().system(' gsutil mb -l $REGION gs://$BUCKET_NAME')

#%% [markdown]
# Finally, validate access to your Cloud Storage bucket by examining its contents:
# 

#%%
get_ipython().system(' gsutil ls -al gs://$BUCKET_NAME')

#%% [markdown]
# ## Part 1. Quickstart for training in AI Platform
# 
# This section of the tutorial walks you through submitting a training job to Cloud
# AI Platform. This job runs sample code that uses Keras to train a deep neural
# network on the United States Census data. It outputs the trained model as a
# [TensorFlow SavedModel
# directory](https://www.tensorflow.org/guide/saved_model#save_and_restore_models)
# in your Cloud Storage bucket.
# 
#%% [markdown]
# ### Get training code and dependencies
# 
# First, download the training code and change the notebook's working directory:

#%%
# Clone the repository of AI Platform samples
get_ipython().system(' git clone --depth 1 https://github.com/GoogleCloudPlatform/cloudml-samples')

# Set the working directory to the sample code directory
get_ipython().run_line_magic('cd', 'cloudml-samples/census/tf-keras')

#%% [markdown]
# Notice that the training code is structured as a Python package in the
# `trainer/` subdirectory:

#%%
# `ls` shows the working directory's contents. The `p` flag adds trailing 
# slashes to subdirectory names. The `R` flag lists subdirectories recursively.
get_ipython().system(' ls -pR')

#%% [markdown]
# Run the following cell to install Python dependencies needed to train the model locally. When you run the training job in AI Platform,
# dependencies are preinstalled based on the [runtime
# version](https://cloud.google.com/ml-engine/docs/tensorflow/runtime-version-list)
# you choose.

#%%
get_ipython().system(' pip install -r requirements.txt')

#%% [markdown]
# ### Train your model locally
# 
# Before training on AI Platform, train the job locally to verify the file
# structure and packaging is correct.
# 
# For a complex or resource-intensive job, you
# may want to train locally on a small sample of your dataset to verify your code.
# Then you can run the job on AI Platform to train on the whole dataset.
# 
# This sample runs a relatively quick job on a small dataset, so the local
# training and the AI Platform job run the same code on the same data.
# 
# Run the following cell to train a model locally:

#%%
# Explicitly tell `gcloud ai-platform local train` to use Python 3 
get_ipython().system(' gcloud config set ml_engine/local_python $(which python3)')

# This is similar to `python -m trainer.task --job-dir local-training-output`
# but it better replicates the AI Platform environment, especially for
# distributed training (not applicable here).
get_ipython().system(' gcloud ai-platform local train   --package-path trainer   --module-name trainer.task   --job-dir local-training-output')

#%% [markdown]
# ### Train your model using AI Platform
# 
# Next, submit a training job to AI Platform. This runs the training module
# in the cloud and exports the trained model to Cloud Storage.
# 
# First, give your training job a name and choose a directory within your Cloud
# Storage bucket for saving intermediate and output files:

#%%
JOB_NAME = 'my_first_keras_job'
JOB_DIR = 'gs://' + BUCKET_NAME + '/keras-job-dir'

#%% [markdown]
# Run the following command to package the `trainer/` directory, upload it to the
# specified `--job-dir`, and instruct AI Platform to run the
# `trainer.task` module from that package.
# 
# The `--stream-logs` flag lets you view training logs in the cell below. You can
# also see logs and other job details in the GCP Console.

#%%
get_ipython().system(' gcloud ai-platform jobs submit training $JOB_NAME   --package-path trainer/   --module-name trainer.task   --region $REGION   --python-version 3.5   --runtime-version 1.13   --job-dir $JOB_DIR   --stream-logs')

#%% [markdown]
# ### Hyperparameter tuning
# 
# You can optionally perform hyperparameter tuning by using the included
# `hptuning_config.yaml` configuration file. This file tells AI Platform to tune the batch size and learning rate for training over multiple trials to maximize accuracy.
# 
# In this example, the training code uses a [TensorBoard
# callback](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/TensorBoard),
# which [creates TensorFlow `Summary`
# `Event`s](https://www.tensorflow.org/api_docs/python/tf/summary/FileWriter#add_summary)
# during training. AI Platform uses these events to track the metric you want to
# optimize. Learn more about [hyperparameter tuning in
# AI Platform Training](https://cloud.google.com/ml-engine/docs/tensorflow/hyperparameter-tuning-overview).

#%%
get_ipython().system(' gcloud ai-platform jobs submit training $JOB_NAME-hpt   --config hptuning_config.yaml   --package-path trainer/   --module-name trainer.task   --region $REGION   --python-version 3.5   --runtime-version 1.13   --job-dir $JOB_DIR   --stream-logs')

#%% [markdown]
# ## Part 2. Quickstart for online predictions in AI Platform
# 
# This section shows how to use AI Platform and your trained model from Part 1
# to predict a person's income bracket from other Census information about them.
#%% [markdown]
# ### Create model and version resources in AI Platform
# 
# To serve online predictions using the model you trained and exported in Part 1,
# create a *model* resource in AI Platform and a *version* resource
# within it. The version resource is what actually uses your trained model to
# serve predictions. This structure lets you adjust and retrain your model many times and
# organize all the versions together in AI Platform. Learn more about [models
# and
# versions](https://cloud.google.com/ml-engine/docs/tensorflow/projects-models-versions-jobs).
# 
# First, name and create the model resource:

#%%
MODEL_NAME = "my_first_keras_model"

get_ipython().system(' gcloud ai-platform models create $MODEL_NAME   --regions $REGION')

#%% [markdown]
# Next, create the model version. The training job from Part 1 exported a timestamped
# [TensorFlow SavedModel
# directory](https://www.tensorflow.org/guide/saved_model#structure_of_a_savedmodel_directory)
# to your Cloud Storage bucket. AI Platform uses this directory to create a
# model version. Learn more about [SavedModel and
# AI Platform](https://cloud.google.com/ml-engine/docs/tensorflow/deploying-models).
# 
# You may be able to find the path to this directory in your training job's logs.
# Look for a line like:
# 
# ```
# Model exported to:  gs://<your-bucket-name>/keras-job-dir/keras_export/1545439782
# ```
# 
# Execute the following command to identify your SavedModel directory and use it to create a model version resource:

#%%
MODEL_VERSION = "v1"

# Get a list of directories in the `keras_export` parent directory
KERAS_EXPORT_DIRS = get_ipython().getoutput('gsutil ls $JOB_DIR/keras_export/')

# Pick the directory with the latest timestamp, in case you've trained
# multiple times
SAVED_MODEL_PATH = KERAS_EXPORT_DIRS[-1]

# Create model version based on that SavedModel directory
get_ipython().system(' gcloud ai-platform versions create $MODEL_VERSION   --model $MODEL_NAME   --runtime-version 1.13   --python-version 3.5   --framework tensorflow   --origin $SAVED_MODEL_PATH')

#%% [markdown]
# ### Prepare input for prediction
# 
# To receive valid and useful predictions, you must preprocess input for prediction in the same way that training data was preprocessed. In a production
# system, you may want to create a preprocessing pipeline that can be used identically at training time and prediction time.
# 
# For this exercise, use the training package's data-loading code to select a random sample from the evaluation data. This data is in the form that was used to evaluate accuracy after each epoch of training, so it can be used to send test predictions without further preprocessing:

#%%
from trainer import util

_, _, eval_x, eval_y = util.load_data()

prediction_input = eval_x.sample(20)
prediction_targets = eval_y[prediction_input.index]

prediction_input

#%% [markdown]
# Notice that categorical fields, like `occupation`,  have already been converted to integers (with the same mapping that was used for training). Numerical fields, like `age`, have been scaled to a
# [z-score](https://developers.google.com/machine-learning/crash-course/representation/cleaning-data). Some fields have been dropped from the original
# data. Compare the prediction input with the raw data for the same examples:

#%%
import pandas as pd

_, eval_file_path = util.download(util.DATA_DIR)
raw_eval_data = pd.read_csv(eval_file_path,
                            names=util._CSV_COLUMNS,
                            na_values='?')

raw_eval_data.iloc[prediction_input.index]

#%% [markdown]
# Export the prediction input to a newline-delimited JSON file:

#%%
import json

with open('prediction_input.json', 'w') as json_file:
  for row in prediction_input.values.tolist():
    json.dump(row, json_file)
    json_file.write('\n')

get_ipython().system(' cat prediction_input.json')

#%% [markdown]
# The `gcloud` command-line tool accepts newline-delimited JSON for online
# prediction, and this particular Keras model expects a flat list of
# numbers for each input example.
# 
# AI Platform requires a different format when you make online prediction requests to the REST API without using the `gcloud` tool. The way you structure
# your model may also change how you must format data for prediction. Learn more
# about [formatting data for online
# prediction](https://cloud.google.com/ml-engine/docs/tensorflow/prediction-overview#prediction_input_data).
#%% [markdown]
# ### Submit the online prediction request
#%% [markdown]
# Use `gcloud` to submit your online prediction request.

#%%
get_ipython().system(' gcloud ai-platform predict   --model $MODEL_NAME   --version $MODEL_VERSION   --json-instances prediction_input.json')

#%% [markdown]
# Since the model's last layer uses a [sigmoid function](https://developers.google.com/machine-learning/glossary/#sigmoid_function) for its activation, outputs between 0 and 0.5 represent negative predictions ("<=50K") and outputs between 0.5 and 1 represent positive ones (">50K").
# 
# Do the predicted income brackets match the actual ones? Run the following cell
# to see the true labels.

#%%
prediction_targets

#%% [markdown]
# ## Part 3. Developing the Keras model from scratch
# 
# At this point, you have trained a machine learning model on AI Platform, deployed the trained model as a version resource on AI Platform, and received online predictions from the deployment. The next section walks through recreating the Keras code used to train your model. It covers the following parts of developing a machine learning model for use with AI Platform:
# 
# * Downloading and preprocessing data
# * Designing and training the model
# * Visualizing training and exporting the trained model
# 
# While this section provides more detailed insight to the tasks completed in previous parts, to learn more about using `tf.keras`, read [TensorFlow's guide to Keras](https://www.tensorflow.org/tutorials/keras). To learn more about structuring code as a training packge for AI Platform, read [Packaging a training application](https://cloud.google.com/ml-engine/docs/tensorflow/packaging-trainer) and reference the [complete training code](https://github.com/GoogleCloudPlatform/cloudml-samples/tree/master/census/tf-keras), which is structured as a Python package.
#%% [markdown]
# ### Import libraries and define constants
# 
# First, import Python libraries required for training:

#%%
import os
from six.moves import urllib
import tempfile

import numpy as np
import pandas as pd
import tensorflow as tf

# Examine software versions
print(__import__('sys').version)
print(tf.__version__)
print(tf.keras.__version__)

#%% [markdown]
# Then, define some useful constants:
# 
# * Information for downloading training and evaluation data
# * Information required for Pandas to interpret the data and convert categorical fields into numeric features
# * Hyperparameters for training, such as learning rate and batch size

#%%
### For downloading data ###

# Storage directory
DATA_DIR = os.path.join(tempfile.gettempdir(), 'census_data')

# Download options.
DATA_URL = 'https://storage.googleapis.com/cloud-samples-data/ml-engine'            '/census/data'
TRAINING_FILE = 'adult.data.csv'
EVAL_FILE = 'adult.test.csv'
TRAINING_URL = '%s/%s' % (DATA_URL, TRAINING_FILE)
EVAL_URL = '%s/%s' % (DATA_URL, EVAL_FILE)

### For interpreting data ###

# These are the features in the dataset.
# Dataset information: https://archive.ics.uci.edu/ml/datasets/census+income
_CSV_COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'gender',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
    'income_bracket'
]

_CATEGORICAL_TYPES = {
    'workclass': pd.api.types.CategoricalDtype(categories=[
        'Federal-gov', 'Local-gov', 'Never-worked', 'Private', 'Self-emp-inc',
        'Self-emp-not-inc', 'State-gov', 'Without-pay'
    ]),
    'marital_status': pd.api.types.CategoricalDtype(categories=[
        'Divorced', 'Married-AF-spouse', 'Married-civ-spouse',
        'Married-spouse-absent', 'Never-married', 'Separated', 'Widowed'
    ]),
    'occupation': pd.api.types.CategoricalDtype([
        'Adm-clerical', 'Armed-Forces', 'Craft-repair', 'Exec-managerial',
        'Farming-fishing', 'Handlers-cleaners', 'Machine-op-inspct',
        'Other-service', 'Priv-house-serv', 'Prof-specialty', 'Protective-serv',
        'Sales', 'Tech-support', 'Transport-moving'
    ]),
    'relationship': pd.api.types.CategoricalDtype(categories=[
        'Husband', 'Not-in-family', 'Other-relative', 'Own-child', 'Unmarried',
        'Wife'
    ]),
    'race': pd.api.types.CategoricalDtype(categories=[
        'Amer-Indian-Eskimo', 'Asian-Pac-Islander', 'Black', 'Other', 'White'
    ]),
    'native_country': pd.api.types.CategoricalDtype(categories=[
        'Cambodia', 'Canada', 'China', 'Columbia', 'Cuba', 'Dominican-Republic',
        'Ecuador', 'El-Salvador', 'England', 'France', 'Germany', 'Greece',
        'Guatemala', 'Haiti', 'Holand-Netherlands', 'Honduras', 'Hong', 'Hungary',
        'India', 'Iran', 'Ireland', 'Italy', 'Jamaica', 'Japan', 'Laos', 'Mexico',
        'Nicaragua', 'Outlying-US(Guam-USVI-etc)', 'Peru', 'Philippines', 'Poland',
        'Portugal', 'Puerto-Rico', 'Scotland', 'South', 'Taiwan', 'Thailand',
        'Trinadad&Tobago', 'United-States', 'Vietnam', 'Yugoslavia'
    ]),
    'income_bracket': pd.api.types.CategoricalDtype(categories=[
        '<=50K', '>50K'
    ])
}

# This is the label (target) we want to predict.
_LABEL_COLUMN = 'income_bracket'

### Hyperparameters for training ###

# This the training batch size
BATCH_SIZE = 128

# This is the number of epochs (passes over the full training data)
NUM_EPOCHS = 20

# Define learning rate.
LEARNING_RATE = .01

#%% [markdown]
# ### Download and preprocess data
#%% [markdown]
# #### Download the data
# 
# Next, define functions to download training and evaluation data. These functions also fix minor irregularities in the data's formatting.

#%%
def _download_and_clean_file(filename, url):
    """Downloads data from url, and makes changes to match the CSV format.
  
    The CSVs may use spaces after the comma delimters (non-standard) or include
    rows which do not represent well-formed examples. This function strips out
    some of these problems.
  
    Args:
      filename: filename to save url to
      url: URL of resource to download
    """
    temp_file, _ = urllib.request.urlretrieve(url)
    with tf.gfile.Open(temp_file, 'r') as temp_file_object:
        with tf.gfile.Open(filename, 'w') as file_object:
            for line in temp_file_object:
                line = line.strip()
                line = line.replace(', ', ',')
                if not line or ',' not in line:
                    continue
                if line[-1] == '.':
                    line = line[:-1]
                line += '\n'
                file_object.write(line)
    tf.gfile.Remove(temp_file)


def download(data_dir):
    """Downloads census data if it is not already present.
  
    Args:
      data_dir: directory where we will access/save the census data
    """
    tf.gfile.MakeDirs(data_dir)

    training_file_path = os.path.join(data_dir, TRAINING_FILE)
    if not tf.gfile.Exists(training_file_path):
        _download_and_clean_file(training_file_path, TRAINING_URL)

    eval_file_path = os.path.join(data_dir, EVAL_FILE)
    if not tf.gfile.Exists(eval_file_path):
        _download_and_clean_file(eval_file_path, EVAL_URL)

    return training_file_path, eval_file_path

#%% [markdown]
# Use those functions to download the data for training and verify that you have CSV files for training and evaluation: 

#%%
training_file_path, eval_file_path = download(DATA_DIR)

# You should see 2 files: adult.data.csv and adult.test.csv
get_ipython().system('ls -l $DATA_DIR')

#%% [markdown]
# Next, load these files using Pandas and examine the data:

#%%
# This census data uses the value '?' for fields (column) that are missing data. 
# We use na_values to find ? and set it to NaN values.
# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html

train_df = pd.read_csv(training_file_path, names=_CSV_COLUMNS, na_values='?')
eval_df = pd.read_csv(eval_file_path, names=_CSV_COLUMNS, na_values='?')

# Here's what the data looks like before we preprocess the data.
train_df.head()

#%% [markdown]
# #### Preprocess the data
# 
# The first preprocessing step removes certain features from the data and
# converts categorical features to numerical values for use with Keras.
# 
# Learn more about [feature engineering](https://developers.google.com/machine-learning/crash-course/representation/feature-engineering) and [bias in data](https://developers.google.com/machine-learning/crash-course/fairness/types-of-bias).

#%%
UNUSED_COLUMNS = ['fnlwgt', 'education', 'gender']


def preprocess(dataframe):
    """Converts categorical features to numeric. Removes unused columns.
  
    Args:
      dataframe: Pandas dataframe with raw data
  
    Returns:
      Dataframe with preprocessed data
    """
    dataframe = dataframe.drop(columns=UNUSED_COLUMNS)

    # Convert integer valued (numeric) columns to floating point
    numeric_columns = dataframe.select_dtypes(['int64']).columns
    dataframe[numeric_columns] = dataframe[numeric_columns].astype('float32')

    # Convert categorical columns to numeric
    cat_columns = dataframe.select_dtypes(['object']).columns
    dataframe[cat_columns] = dataframe[cat_columns].apply(lambda x: x.astype(
        _CATEGORICAL_TYPES[x.name]))
    dataframe[cat_columns] = dataframe[cat_columns].apply(lambda x: x.cat.codes)
    return dataframe


prepped_train_df = preprocess(train_df)
prepped_eval_df = preprocess(eval_df)

#%% [markdown]
# Run the following cell to see how preprocessing changed the data. Notice in particular that `income_bracket`, the label that you're training the model to predict, has changed from `<=50K` and `>50K` to `0` and `1`:

#%%
prepped_train_df.head()

#%% [markdown]
# Next, separate the data into features ("x") and labels ("y"), and reshape the label arrays into a format for use with `tf.data.Dataset` later:

#%%
# Split train and test data with labels.
# The pop() method will extract (copy) and remove the label column from the dataframe
train_x, train_y = prepped_train_df, prepped_train_df.pop(_LABEL_COLUMN)
eval_x, eval_y = prepped_eval_df, prepped_eval_df.pop(_LABEL_COLUMN)

# Reshape label columns for use with tf.data.Dataset
train_y = np.asarray(train_y).astype('float32').reshape((-1, 1))
eval_y = np.asarray(eval_y).astype('float32').reshape((-1, 1))

#%% [markdown]
# Scaling training data so each numerical feature column has a mean of 0 and a standard deviation of 1 [can improve your model](https://developers.google.com/machine-learning/crash-course/representation/cleaning-data).
# 
# In a production system, you may want to save the means and standard deviations from your training set and use them to perform an identical transformation on test data at prediction time. For convenience in this exercise, temporarily combine the training and evaluation data to scale all of them:

#%%
def standardize(dataframe):
    """Scales numerical columns using their means and standard deviation to get
    z-scores: the mean of each numerical column becomes 0, and the standard
    deviation becomes 1. This can help the model converge during training.
  
    Args:
      dataframe: Pandas dataframe
  
    Returns:
      Input dataframe with the numerical columns scaled to z-scores
    """
    dtypes = list(zip(dataframe.dtypes.index, map(str, dataframe.dtypes)))
    # Normalize numeric columns.
    for column, dtype in dtypes:
        if dtype == 'float32':
            dataframe[column] -= dataframe[column].mean()
            dataframe[column] /= dataframe[column].std()
    return dataframe


# Join train_x and eval_x to normalize on overall means and standard
# deviations. Then separate them again.
all_x = pd.concat([train_x, eval_x], keys=['train', 'eval'])
all_x = standardize(all_x)
train_x, eval_x = all_x.xs('train'), all_x.xs('eval')

#%% [markdown]
# Finally, examine some of your fully preprocessed training data:

#%%
# Verify dataset features
# Note how only the numeric fields (not categorical) have been standardized
train_x.head()

#%% [markdown]
# ### Design and train the model
#%% [markdown]
# #### Create training and validation datasets
# 
# Create an input function to convert features and labels into a
# [`tf.data.Dataset`](https://www.tensorflow.org/guide/datasets) for training or evaluation:

#%%
def input_fn(features, labels, shuffle, num_epochs, batch_size):
    """Generates an input function to be used for model training.
  
    Args:
      features: numpy array of features used for training or inference
      labels: numpy array of labels for each example
      shuffle: boolean for whether to shuffle the data or not (set True for
        training, False for evaluation)
      num_epochs: number of epochs to provide the data for
      batch_size: batch size for training
  
    Returns:
      A tf.data.Dataset that can provide data to the Keras model for training or
        evaluation
    """
    if labels is None:
        inputs = features
    else:
        inputs = (features, labels)
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(features))

    # We call repeat after shuffling, rather than before, to prevent separate
    # epochs from blending together.
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    return dataset

#%% [markdown]
# Next, create these training and evaluation datasets.Use the `NUM_EPOCHS`
# and `BATCH_SIZE` hyperparameters defined previously to define how the training
# dataset provides examples to the model during training. Set up the validation
# dataset to provide all its examples in one batch, for a single validation step
# at the end of each training epoch.

#%%
# Pass a numpy array by using DataFrame.values
training_dataset = input_fn(features=train_x.values, 
                    labels=train_y, 
                    shuffle=True, 
                    num_epochs=NUM_EPOCHS, 
                    batch_size=BATCH_SIZE)

num_eval_examples = eval_x.shape[0]

# Pass a numpy array by using DataFrame.values
validation_dataset = input_fn(features=eval_x.values, 
                    labels=eval_y, 
                    shuffle=False, 
                    num_epochs=NUM_EPOCHS, 
                    batch_size=num_eval_examples)                

#%% [markdown]
# #### Design a Keras Model
#%% [markdown]
# Design your neural network using the [Keras Sequential API](https://www.tensorflow.org/guide/keras#sequential_model).
# 
# This deep neural network (DNN) has several hidden layers, and the last layer uses a sigmoid activation function to output a value between 0 and 1:
# 
# * The input layer has 100 units using the ReLU activation function.
# * The hidden layer has 75 units using the ReLU activation function.
# * The hidden layer has 50 units using the ReLU activation function.
# * The hidden layer has 25 units using the ReLU activation function.
# * The output layer has 1 units using a sigmoid activation function.
# * The optimizer uses the binary cross-entropy loss function, which is appropriate for a binary classification problem like this one.
# 
# Feel free to change these layers to try to improve the model:

#%%
def create_keras_model(input_dim, learning_rate):
    """Creates Keras Model for Binary Classification.
  
    Args:
      input_dim: How many features the input has
      learning_rate: Learning rate for training
  
    Returns:
      The compiled Keras model (still needs to be trained)
    """
    Dense = tf.keras.layers.Dense
    model = tf.keras.Sequential(
      [
          Dense(100, activation=tf.nn.relu, kernel_initializer='uniform',
                  input_shape=(input_dim,)),
          Dense(75, activation=tf.nn.relu),
          Dense(50, activation=tf.nn.relu),
          Dense(25, activation=tf.nn.relu),
          Dense(1, activation=tf.nn.sigmoid)
      ])
    # Custom Optimizer:
    # https://www.tensorflow.org/api_docs/python/tf/train/RMSPropOptimizer
    optimizer = tf.keras.optimizers.RMSprop(
        lr=learning_rate)

    # Compile Keras model
    model.compile(
        loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

#%% [markdown]
# Next, create the Keras model object and examine its structure:

#%%
num_train_examples, input_dim = train_x.shape
print('Number of features: {}'.format(input_dim))
print('Number of examples: {}'.format(num_train_examples))

keras_model = create_keras_model(
    input_dim=input_dim,
    learning_rate=LEARNING_RATE)

# Take a detailed look inside the model
keras_model.summary()

#%% [markdown]
# #### Train and evaluate the model
# 
# Define a learning rate decay to encourage model paramaters to make smaller
# changes as training goes on:

#%%
# Setup Learning Rate decay.
lr_decay_cb = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: LEARNING_RATE + 0.02 * (0.5 ** (1 + epoch)),
    verbose=True)

# Setup TensorBoard callback.
tensorboard_cb = tf.keras.callbacks.TensorBoard(
      os.path.join(JOB_DIR, 'keras_tensorboard'),
      histogram_freq=1)

#%% [markdown]
# Finally, train the model. Provide the appropriate `steps_per_epoch` for the
# model to train on the entire training dataset (with `BATCH_SIZE` examples per step) during each epoch. And instruct the model to calculate validation
# accuracy with one big validation batch at the end of each epoch.
# 

#%%
history = keras_model.fit(training_dataset, 
                          epochs=NUM_EPOCHS, 
                          steps_per_epoch=int(num_train_examples/BATCH_SIZE), 
                          validation_data=validation_dataset, 
                          validation_steps=1, 
                          callbacks=[lr_decay_cb, tensorboard_cb],
                          verbose=1)

#%% [markdown]
# ### Visualize training and export the trained model
#%% [markdown]
# #### Visualize training
# 
# Import `matplotlib` to visualize how the model learned over the training period.

#%%
get_ipython().system(' pip install matplotlib')

from matplotlib import pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

#%% [markdown]
# Plot the model's loss (binary cross-entropy) and accuracy, as measured at the
# end of each training epoch:

#%%
# Visualize History for Loss.
plt.title('Keras model loss')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='upper right')
plt.show()

# Visualize History for Accuracy.
plt.title('Keras model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.legend(['training', 'validation'], loc='lower right')
plt.show()

#%% [markdown]
# Over time, loss decreases and accuracy increases. But do they converge to a
# stable level? Are there big differences between the training and validation
# metrics (a sign of overfitting)?
# 
# Learn about [how to improve your machine learning
# model](https://developers.google.com/machine-learning/crash-course/). Then, feel
# free to adjust hyperparameters or the model architecture and train again.
#%% [markdown]
# #### Export the model for serving
# 
# Use
# [tf.contrib.saved_model.save_keras_model](https://www.tensorflow.org/api_docs/python/tf/contrib/saved_model/save_keras_model) to export a TensorFlow SavedModel directory. This is the format that Cloud
# AI Platform requires when you [create a model version
# resource](https://cloud.google.com/ml-engine/docs/tensorflow/deploying-models#create_a_model_version).
# 
# Since not all optimizers can be exported to the SavedModel format, you may see
# warnings during the export process. As long you successfully export a serving
# graph, AI Platform can used the SavedModel to serve predictions.

#%%
# Export the model to a local SavedModel directory 
export_path = tf.contrib.saved_model.save_keras_model(keras_model, 'keras_export')
print("Model exported to: ", export_path)

#%% [markdown]
# You may export a SavedModel directory to your local filesystem or to Cloud
# Storage, as long as you have the necessary permissions. In your current
# environment, you granted access to Cloud Storage by authenticating your GCP account and setting the `GOOGLE_APPLICATION_CREDENTIALS` environment variable.
# AI Platform training jobs can also export directly to Cloud Storage, because
# AI Platform service accounts [have access to Cloud Storage buckets in their own
# project](https://cloud.google.com/ml-engine/docs/tensorflow/working-with-cloud-storage).
# 
# Try exporting directly to Cloud Storage:

#%%
# Export the model to a SavedModel directory in Cloud Storage
export_path = tf.contrib.saved_model.save_keras_model(keras_model, JOB_DIR + '/keras_export')
print("Model exported to: ", export_path)

#%% [markdown]
# You can now deploy this model to AI Platform and serve predictions by
# following the steps from Part 2.
#%% [markdown]
# ## Cleaning up
# 
# To clean up all GCP resources used in this project, you can [delete the GCP
# project](https://cloud.google.com/resource-manager/docs/creating-managing-projects#shutting_down_projects) you used for the tutorial.
# 
# Alternatively, you can clean up individual resources by running the following
# commands:

#%%
# Delete model version resource
get_ipython().system(' gcloud ai-platform versions delete $MODEL_VERSION --quiet --model $MODEL_NAME ')

# Delete model resource
get_ipython().system(' gcloud ai-platform models delete $MODEL_NAME --quiet')

# Delete Cloud Storage objects that were created
get_ipython().system(' gsutil -m rm -r $JOB_DIR')

# If the training job is still running, cancel it
get_ipython().system(' gcloud ai-platform jobs cancel $JOB_NAME --quiet --verbosity critical')

#%% [markdown]
# If your Cloud Storage bucket doesn't contain any other objects and you would like to delete it, run `gsutil rm -r gs://$BUCKET_NAME`.
#%% [markdown]
# ## What's next?
# 
# * View the [complete training
# code](https://github.com/GoogleCloudPlatform/cloudml-samples/tree/master/census/tf-keras) used in this guide, which structures the code to accept custom
# hyperparameters as command-line flags.
# * Read about [packaging
# code](https://cloud.google.com/ml-engine/docs/tensorflow/packaging-trainer) for an AI Platform training job.
# * Read about [deploying a
# model](https://cloud.google.com/ml-engine/docs/tensorflow/deploying-models) to serve predictions.

