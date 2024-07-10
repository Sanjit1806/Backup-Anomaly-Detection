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

p1 = LocalOutlierFactor(
    n_neighbors=50, # number of nearest neighbors considered
    contamination=0.05, # proportion of the outliers in the data
    metric='minkowski', # distance metric in higher dimensional space
    novelty=True
)

p1.fit(scaled_df)

p2 = Pipeline(
    steps=[('preprocessor', MinMaxScaler()), # Normalize the data
           ('model', DBSCAN(eps=0.25))]
) # DBSCAN Model

p2.fit(train_data)

p3 = Pipeline(steps=[('preprocessor', MinMaxScaler()), # Normalize the data
 ('model', IsolationForest(
    contamination=0.05, # Proportion of outliers in the data
    random_state=42, # For code reproductability
    n_estimators=100 # Number of decision tree estimators involved.
))])

p3.fit(train_data)

joblib.dump(model, 'model.pkl')
