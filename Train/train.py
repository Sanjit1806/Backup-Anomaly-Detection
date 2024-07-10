import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from data import fetch_train_data, cols
import joblib

scaler = MinMaxScaler()
data = fetch_train_data()
scaler.fit(data.values)
scaled_data = scaler.transform(data.values)
scaled_df = pd.DataFrame(scaled_data, columns=cols+['size_to_nt'])

model = LocalOutlierFactor(
    n_neighbors=50, # number of nearest neighbors considered
    contamination=0.05, # proportion of the outliers in the data
    metric='minkowski', # distance metric in higher dimensional space
    novelty=True
)

model.fit(scaled_df)

joblib.dump(model, 'model.pkl')
