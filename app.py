import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

df=pd.read_csv("data/Churn_Modelling.csv")

df=df.drop(columns=['RowNumber','CustomerId', 'Surname'])


le=LabelEncoder()
df['Geography']=le.fit_transform(df['Geography'])
df['Gender']=le.fit_transform(df['Gender'])

model=joblib.load('models/model_v1.pkl')

X=df.drop(columns=['Exited'])

prediction=model.predict(X)

print("Prediction : ",prediction)
