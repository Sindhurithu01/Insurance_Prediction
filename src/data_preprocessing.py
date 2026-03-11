import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
def load_and_Split_data():
    data=pd.read_csv("C:\\Users\\akhil\\OneDrive\\Documents\\Project\\Insurance_Prediction\\data\\raw\\insurance_data.csv")
    X=data[['Age','Annual_Income_LPA','Policy_Term_Years','Sum_Assured_Lakhs']]
    y=data['Annual_Premium_Thousands']
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
    return X_train,X_test,y_train,y_test

 
