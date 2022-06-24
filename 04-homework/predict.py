import pickle
import pandas as pd
import sys

with open('model.bin', 'rb') as f_in:
    dv, lr = pickle.load(f_in)
    
    
def read_data(filename):
    df = pd.read_parquet(filename)
    categorical = ['PUlocationID', 'DOlocationID']

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df[categorical]  
    
def prepare_features(ride):
    features = ride.to_dict(orient='records')
    return features

def predict(features):
    X = dv.transform(features)
    preds = lr.predict(X)
    return preds

def run(year = "2021", month = "02"):
    df = read_data('https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_' + year + '-' + month +'.parquet')
    features = prepare_features(df)
    results = predict(features)
    print(results.mean())

if __name__ == "__main__":
    if len(sys.argv) > 1:
        a = str(sys.argv[1])
        b = str(sys.argv[2])
        run(a, b)
    else: 
        run()