import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from prefect import flow, task, get_run_logger

import datetime

import pickle

@task
def get_paths(date):
    format = "%Y-%m-%d"
    dt = datetime.datetime.strptime(date, format).replace(day=1)
    last = dt - datetime.timedelta(days=1)
    lastMonth = last.strftime("%m")
    lastYear = last.strftime("%Y")
    penult = last.replace(day=1)- datetime.timedelta(days=1)
    penultMonth = penult.strftime("%m")
    penultYear = penult.strftime("%Y")
    
    s = "../data/fhv_tripdata_"
    s1 = s + str(penultYear) + "-" + str(penultMonth) + ".parquet"
    s2 = s + str(lastYear) + "-" + str(lastMonth) + ".parquet"
    
    return s1, s2
    

@task
def read_data(path):
    df = pd.read_parquet(path)
    return df

@task
def prepare_features(df, categorical, train=True):
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        print(f"The mean duration of training is {mean_duration}")
    else:
        print(f"The mean duration of validation is {mean_duration}")
    
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

@task
def train_model(df, categorical):

    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts) 
    y_train = df.duration.values

    print(f"The shape of X_train is {X_train.shape}")
    print(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    print(f"The MSE of training is: {mse}")
    return lr, dv

@task
def run_model(df, categorical, dv, lr):
    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts) 
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    print(f"The MSE of validation is: {mse}")
    return

@flow
def main(date=None):
    if date == None:
        #if there is the most recent data uncomment the following:
        #date = datetime.datetime.today().strftime('%Y-%m-%d')
        date = "2021-03-15"
    train_path, val_path = get_paths(date).result()

    categorical = ['PUlocationID', 'DOlocationID']

    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical)

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical, False)

    # train the model
    lr, dv = train_model(df_train_processed, categorical).result()
    run_model(df_val_processed, categorical, dv, lr)
    with open("models/model-" + date + ".b", "wb") as f_out:
        pickle.dump(lr, f_out)
    with open("models/dv-" + date + ".b", "wb") as f_out:
        pickle.dump(dv, f_out)

#main(date="2021-08-15")

from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import CronSchedule

DeploymentSpec(
    name="model_training",
    flow=main,
    schedule=CronSchedule(
        cron="0 0 15 9 0",
        timezone="America/New_York"),
    tags=["ml"],
)