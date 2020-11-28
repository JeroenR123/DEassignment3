# %%
import json
import kfp
import kfp.dsl as dsl
import kfp.components as comp
from kfp.components import OutputPath

# %%
"""
#### Pipeline Configurations
"""

# %%
# replace this value with the value of the KFP host name
KFP_HOST_NAME = 'https://527bd6f955d1f721-dot-us-central2.pipelines.googleusercontent.com/'
# change caching policy of KFP run (useful in debug/dev mode)
disable_cache = False
#number of days hold out for validation
validation_days = 100

# %%
client = kfp.Client(host=KFP_HOST_NAME)

# %%
"""
#### Component - Download raw data
"""

# %%
def download_raw_data(raw_data_path: str) -> str:
    '''load data from local storage'''
    import pandas as pd
    import numpy as np
    from google.cloud import storage
    import re

    
    """Downloads a blob from the bucket."""
    bucket_name = "bucket_de"
    source_blob_name = "raw/raw_data2.csv"
    destination_file_name = "tmp/raw_data2.csv"

    storage_client = storage.Client()
    
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print('Downloaded data...')

    #import wget
    #wget -O bestand --post-data="variabele=waarde&variabele=waarde&...." http://projects.knmi.nl/klimatologie/daggegevens/getdata_dag.cgi
    #df = pd.read_csv('gs://bucket_de/raw/raw_data2,csv', error_bad_lines = False)
    df =  pd.read_csv("tmp/raw_data2.csv", error_bad_lines=False)

    
    def dfreplace(df, *args, **kwargs):
        s = pd.Series(df.values.flatten())
        s = s.str.replace(*args, **kwargs)
        return pd.DataFrame(s.values.reshape(df.shape), df.index, df.columns)
    
    weather_df = dfreplace(df, ',', '')
    
    for i in weather_df.columns:
        weather_df[i] = weather_df[i].astype(str)
        weather_df[i][weather_df[i].apply(lambda i: True if re.search('^\s*$', str(i)) else False)]=np.NaN

    print(weather_df.head())
    print('trying to write to GS')
    weather_df.to_parquet(raw_data_path, compression='GZIP')
    print('Done!')
    return raw_data_path

# %%
# create a KFP component
download_raw_data_op = comp.create_component_from_func(
    download_raw_data, output_component_file='download_raw_data.yaml', packages_to_install=['fastparquet', 'fsspec', 'gcsfs', "google-cloud-storage"])

# %%
"""
#### Component - Feature processing
"""

# %%
def feature_processing(raw_data_path: str, feature_data_path: str) -> str:
    '''calculate features for our machine learning model'''
    import pandas as pd
    from datetime import datetime

    # read dataframe
    weather_df = pd.read_parquet(raw_data_path)
    
    # create empty df to store feature
    weather_features_df = weather_df
    
    #create variables for years, months and days
    weather_features_df['YYYY'] = weather_features_df['YYYYMMDD,'].str.slice(0,4) #create a variable for years
    weather_features_df['MM'] = weather_features_df['YYYYMMDD,'].str.slice(4,6)#create a variable for months
    weather_features_df['DD'] = weather_features_df['YYYYMMDD,'].str.slice(6,8)
    for i in weather_features_df.columns:
            weather_features_df[i] = weather_features_df[i].astype(float, errors= 'ignore') 
    weather_features_df = weather_features_df.drop('YYYYMMDD,', axis=1)
    
    weather_features_df['TG_future'] = weather_features_df['TG,'].shift(periods = -1)
    
    #remove irrelevant features manually
    weather_features_df = weather_features_df.drop(columns = ['STN,','EV24', 'NG,', 'TN,', 'TNH,', 'TX,', 'TXH,', 'T10N,', 'T10NH,'])
    weather_features_df = weather_features_df.dropna()

    weather_features_df.to_parquet(feature_data_path, compression='GZIP')
    features_numbers = len(weather_features_df.columns) - 1
    print('Writing %s features' % (features_numbers))
    print('Done!')
    
    return feature_data_path

# %%
# create a KFP component
feature_processing_op = comp.create_component_from_func(
    feature_processing, output_component_file='feature_processing.yaml', packages_to_install=['fastparquet', 'fsspec', 'gcsfs', 'google-cloud-storage'])

# %%
"""
#### Component - Train weather model regression
"""

# %%
def train_regression_weather_model(feature_data_path: str, regression_weather_model_path: str, validation_days: int) -> str:
    '''train a regression model with default parameters'''
    import pandas as pd
    import _pickle as cPickle # save ML model
    from google.cloud import storage # save the model to GCS
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from urllib.parse import urlparse
    
    # read dataframe
    weather_df = pd.read_parquet(feature_data_path)
    # holdout latest datelines used for validation
    weather_features_df = weather_df.drop(weather_df.tail(validation_days).index, inplace=False, axis=0)    
    # get x and y
    x_train, y_train = weather_features_df.drop('TG_future', axis=1), weather_features_df['TG_future']
    # split the data for initial testing
    X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_train, test_size=0.2,random_state=1)
    
    # train the model
    print('Training regression model')
    print('Shape of X: %s, %s' % (len(x_train), len(x_train.columns)))
    regression_model = LinearRegression()
    regression_model.fit(X_train, Y_train)
    
    # some initial testing
    predictions = regression_model.predict(X_test)
    print('mean absolute error without optimization: %s' % mean_absolute_error(Y_test, predictions))
    print('mean squared error without optimization is: %s' % mean_squared_error(Y_test, predictions)) 
    
    # write out output
    # save the model into temp
    with open('/tmp/model.pickle', 'wb') as f:
        cPickle.dump(regression_model, f, -1)
        
    # get client and write to GCS
    # parse model write path for GS
    parse = urlparse(url=regression_weather_model_path, allow_fragments=False)
    if parse.path[0] =='/':
        model_path = parse.path[1:]
        
    client = storage.Client()
    bucket = client.get_bucket(parse.netloc)
    blob = bucket.blob(model_path)
    blob.upload_from_filename('/tmp/model.pickle')
    
    return regression_weather_model_path

# %%
# create a KFP component
train_regression_weather_model_op = comp.create_component_from_func(
    train_regression_weather_model, output_component_file='train_regression_weather_model.yaml', packages_to_install=['scikit-learn', 'fastparquet', 'fsspec', 'gcsfs', 'google-cloud-storage'])

# %%
"""
#### Component - Train Random Forest weather model
"""

# %%
def train_rf_weather_model(feature_data_path: str, rf_weather_model_path: str, validation_days: int) -> str:
    '''train a random forest model with default parameters'''
    import json
    import pandas as pd
    import _pickle as cPickle # save ML model
    from google.cloud import storage # save the model to GCS
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn.model_selection import train_test_split
    from urllib.parse import urlparse
    
    # read dataframe
    weather_df = pd.read_parquet(feature_data_path)
    # holdout latest datelines used for validation
    weather_features_df = weather_df.drop(weather_df.tail(validation_days).index, inplace=False, axis=0)
    # get x and y
    x_train, y_train = weather_features_df.drop('TG_future', axis=1), weather_features_df['TG_future']
    # split the data for initial testing
    X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_train, test_size=0.2,random_state=1)
    
    # train the model
    print('Training random forest model')
    print('Shape of X: %s, %s' % (len(x_train), len(x_train.columns)))
    model_rf = RandomForestClassifier(random_state = 42, criterion = 'entropy',
                                        max_depth = 3, min_samples_split = 0.1)

    model_rf.fit(X_train, Y_train)
    
    # some initial testing
    predictions = model_rf.predict(X_test)
    print('mean absolute error without optimization: %s' % mean_absolute_error(Y_test, predictions))
    print('mean squared error without optimization is: %s' % mean_squared_error(Y_test, predictions)) 
    
    temp_model_path = '/tmp/model.pickle'
    
    # write out output
    # save the model into temp
    with open(temp_model_path, 'wb') as f:
        cPickle.dump(model_rf, f, -1)
        
    # get client and write to GCS
    # parse model write path for GS
    parse = urlparse(url=rf_weather_model_path, allow_fragments=False)
    
    if parse.path[0] =='/':
        model_path = parse.path[1:]
    client = storage.Client()
    bucket = client.get_bucket(parse.netloc)
    model = bucket.blob(model_path)
    model.upload_from_filename(temp_model_path)
    
    return rf_weather_model_path

# %%
# create a KFP component
train_rf_weather_model_op = comp.create_component_from_func(
    train_rf_weather_model, output_component_file='train_rf_weather_model.yaml', packages_to_install=['scikit-learn', 'fastparquet', 'fsspec', 'gcsfs', 'google-cloud-storage'])

# %%
"""
#### Component - Evaluate the models
"""

# %%
def eval_models(feature_data_path: str, regression_weather_model_path, rf_weather_model_path: str, validation_days: int) -> None:
    '''Evaluate different models on holdout dataset to see which model performs the best'''
    import json
    import pandas as pd
    from io import BytesIO
    import _pickle as cPickle # save ML model
    from google.cloud import storage # save the model to GCS
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn.model_selection import train_test_split
    from urllib.parse import urlparse
    from collections import namedtuple
    
    # read dataframe
    weather_df = pd.read_parquet(feature_data_path)
    # holdout latest datelines used for validation
    weather_features_df = weather_df.drop(weather_df.tail(validation_days).index, inplace=False, axis=0)#validation set
    
    weather_validation_df = weather_df.tail(validation_days)
    
    x_val, y_val = weather_validation_df.drop('TG_future', axis=1), weather_validation_df['TG_future']
    
    def get_mae(model_path):
        '''this function evaluates a model on the validation dataset given just the model path'''
        parse = urlparse(url=model_path, allow_fragments=False)

        if parse.path[0] =='/':
            model_path = parse.path[1:]

        client = storage.Client()
        bucket = client.get_bucket(parse.netloc)
        blob = bucket.get_blob(model_path)
        if blob is None:
            raise AttributeError('No files to download') 
        model_bytestream = BytesIO(blob.download_as_string())
        model = cPickle.load(model_bytestream)
        predictions = model.predict(x_val)
        return mean_absolute_error(y_val, predictions)
    
    Models = namedtuple('Model', 'type score path')
    m_list = list()
    
    regression_mae = get_mae(regression_weather_model_path)
    m_list.append(Models('regression', regression_mae, regression_weather_model_path))
    
    rf_mae = get_mae(rf_weather_model_path)
    m_list.append(Models('rf', rf_mae, rf_weather_model_path))
    
    max_score = max([model.score for model in m_list])
    max_score_index = [model.score for model in m_list].index(max_score)
    print('Best Model: ', m_list[max_score_index])
    path = m_list[max_score_index].path
    return path

# %%
# create a KFP component
eval_models_op = comp.create_component_from_func(
    eval_models, output_component_file='eval_models.yaml', packages_to_install=['scikit-learn', 'fastparquet', 'fsspec', 'gcsfs', 'google-cloud-storage'])

# %%
"""
#### Create and run KubeFlow pipeline
"""

# %%
@dsl.pipeline(
  name='Weather regression',
  description='Predicting temperature in Eindoven using linear regression and random forest'
)
def weather_pipeline(raw_data_path, feature_data_path, regression_weather_model_path, rf_weather_model_path, disable_cache, validation_days):

  download_raw_data_task = download_raw_data_op(raw_data_path)
  feature_processing_task = feature_processing_op(download_raw_data_task.output, feature_data_path)
  train_regression_weather_model_task = train_regression_weather_model_op(feature_processing_task.output, regression_weather_model_path, validation_days)
  train_rf_weather_model_task = train_rf_weather_model_op(feature_processing_task.output, rf_weather_model_path, validation_days)
  eval_models_task = eval_models_op(feature_processing_task.output, train_regression_weather_model_task.output, train_rf_weather_model_task.output, validation_days)

  if disable_cache:
      download_raw_data_task.execution_options.caching_strategy.max_cache_staleness = "P0D"
      feature_processing_task.execution_options.caching_strategy.max_cache_staleness = "P0D"
      train_regression_weather_model_task.execution_options.caching_strategy.max_cache_staleness = "P0D"
      train_rf_weather_model_task.execution_options.caching_strategy.max_cache_staleness = "P0D"
      eval_models_task.execution_options.caching_strategy.max_cache_staleness = "P0D"
    
# Specify argument values for pipeline run.
arguments = {'raw_data_path': 'gs://bucket_de/raw/weater.parquet',
            'feature_data_path': 'gs://bucket_de/raw/feature_data/weater_features.parquet',
            'regression_weather_model_path': 'gs://bucket_de/model_store/vanilla/vanilla_rf.pickle',
            'rf_weather_model_path': 'gs://bucket_de/model_store/rf/rf_rf.pickle',
            'disable_cache': disable_cache,
            'validation_days': validation_days,
            }


    
# Create a pipeline run, using the client you initialized in a prior step.
#client.create_run_from_pipeline_func(weather_pipeline, arguments=arguments)
if __name__ == '__main__':
    kfp.compiler.Compiler().compile(p, __file__ + '.yaml')

# %%
