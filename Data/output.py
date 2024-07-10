import joblib
import pandas as pd
from data import fetch_new_data, cols
from train import scaler

model = joblib.load('model.pkl')
print(model.predict(pd.DataFrame(
    data=scaler.transform(fetch_new_data().values),
    columns=cols+['size_to_nt']
)))
