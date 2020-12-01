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
KFP_HOST_NAME = 'https://13e48231a6e5f8f-dot-us-central2.pipelines.googleusercontent.com/'
# change caching policy of KFP run (useful in debug/dev mode)
disable_cache = False
#number of days hold out for validation
mltype = 'rf'

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
    from datetime import date, timedelta

    yesterday = date.today() - timedelta(days=1)

    # YYYYMMDD
    yesterday = yesterday.strftime("%Y%m%d")
    print("d1 =", yesterday)
    
    API_URL = 'http://projects.knmi.nl/klimatologie/daggegevens/getdata_dag.cgi'
    API_KEY = "stns=370"
    # 370:         5.377       51.451      22.60  EINDHOVEN
    import urllib3
    import json
    
    with open("raw_data.txt", "w") as dt:
        import requests
        stations = [370]
        params = {
            "stns": "370",
            "start": yesterday,
        }
        print(params)

        url = 'http://projects.knmi.nl/klimatologie/daggegevens/getdata_dag.cgi'
        myobj = API_KEY

        r = requests.post(url, data = params)
        dt.write(r.text)

    with open('raw_data.txt', 'r') as fin:
        data = fin.read().splitlines(True)
    with open('edit.txt', 'w') as fout:
        fout.writelines(data[48:])
    with open('edit.txt', 'r') as fin:
        file = fin.readlines()
    
    import csv
    import re
    with open('raw_data.csv', 'w') as dt:
        csvv = csv.writer(dt)
        file[0] = file[0].replace('# ','')
        for row in file:
            row = re.sub(r'(?<=[,])(?=[^\s])', r' ', row)
            csvv.writerow(row.split())

    
    df =  pd.read_csv("raw_data.csv", error_bad_lines=False)
    
    def dfreplace(df, *args, **kwargs):
        s = pd.Series(df.values.flatten())
        s = s.str.replace(*args, **kwargs)
        return pd.DataFrame(s.values.reshape(df.shape), df.index, df.columns)
    
    df = df.drop([0], axis = 0)
    weather_df = dfreplace(df, ',', '')
     
    for i in weather_df.columns:
        weather_df[i] = weather_df[i].astype(str)
        weather_df[i][weather_df[i].apply(lambda i: True if re.search('^\s*$', str(i)) else False)]=np.NaN
    print(weather_df.columns)
    print(weather_df)
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
def feature_processing(raw_data_path: str, new_feature_data_path: str) -> str:
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
    weather_features_df['TG,'] = weather_features_df['TG,'].div(10)
    weather_features_df['TG_future'] = weather_features_df['TG,']
    
    #remove irrelevant features manually
    weather_features_df = weather_features_df.drop(columns = ['STN,','EV24,', 'NG,', 'TN,', 'TNH,', 'TX,', 'TXH,', 'T10N,', 'T10NH,', 'UNH'])
    weather_features_df = weather_features_df.dropna()

    weather_features_df.to_parquet(new_feature_data_path, compression='GZIP')
    features_numbers = len(weather_features_df.columns) - 1
    print('Writing %s features' % (features_numbers))
    print(weather_features_df)
    print('Done!')
    
    return new_feature_data_path

# %%
# create a KFP component
feature_processing_op = comp.create_component_from_func(
    feature_processing, output_component_file='feature_processing.yaml', packages_to_install=['fastparquet', 'fsspec', 'gcsfs', 'google-cloud-storage'])

# %%
"""
#### Component - Predict model
"""

# %%
def predict_weather_model(new_feature_data_path: str, mltype: str, prediction_path: str) -> str:
    import json
    import pandas as pd
    import _pickle as cPickle # save ML model
    from google.cloud import storage # save the model to GCS
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn.model_selection import train_test_split
    from urllib.parse import urlparse
    from io import BytesIO
    import numpy as np
    if mltype == 'rf':
        model_path = 'gs://bucket_de/model_store/rf/model.pkl'
    elif mltype == 'reg':
        model_path = 'gs://bucket_de/model_store/vanilla/model.pkl'
    


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
    weather_features_df = pd.read_parquet(new_feature_data_path)
    x = weather_features_df.drop('TG_future', axis=1)
    print(x)
    predictions = model.predict(x)
    predictions = pd.DataFrame(data= {'prediction': predictions})
    print('the weather today will be %s' % (predictions))
    predictions.to_parquet(prediction_path, compression='GZIP')
    return prediction_path

# %%
# create a KFP component
predict_weather_model_op = comp.create_component_from_func(
    predict_weather_model, output_component_file='train_regression_weather_model.yaml', packages_to_install=['scikit-learn', 'fastparquet', 'fsspec', 'gcsfs', 'google-cloud-storage'])

# %%
"""
#### Create and run KubeFlow pipeline
"""

# %%
@dsl.pipeline(
  name='Predict weather',
  description='Predicting temperature in Eindoven using linear regression and random forest'
)
def predict_pipeline(raw_data_path, new_feature_data_path, prediction_path, mltype, disable_cache):

    download_raw_data_task = download_raw_data_op(raw_data_path)
    feature_processing_task = feature_processing_op(download_raw_data_task.output, new_feature_data_path)
    prediction_task = predict_weather_model_op(feature_processing_task.output, mltype, prediction_path)
    
    if disable_cache:
        download_raw_data_task.execution_options.caching_strategy.max_cache_staleness = "P0D"
        feature_processing_task.execution_options.caching_strategy.max_cache_staleness = "P0D"
        prediction_task.execution_options.caching_strategy.max_cache_staleness = "P0D"


# Specify argument values for pipeline run.
arguments = {'raw_data_path': 'gs://bucket_de/new_raw/weater.parquet',
            'new_feature_data_path': 'gs://bucket_de/raw/new_data/weater_features.parquet',
            'prediction_path' :'gs://bucket_de/model_store/prediction/prediction.parquet',
            'disable_cache': disable_cache,
            'mltype' : mltype
            }


    
# Create a pipeline run, using the client you initialized in a prior step.
client.create_run_from_pipeline_func(predict_pipeline, arguments=arguments)
"""if __name__ == '__main__':
    kfp.compiler.Compiler().compile(p, __file__ + '.yaml')"""

# %%
"""
#ipynb-py-convert predict_cont_pipeline.ipynb predict_cont_pipeline.py

#dsl-compile --py './predict_cont_pipeline' --output './predict_cont_pipeline.tar.gz'
"""

# %%
