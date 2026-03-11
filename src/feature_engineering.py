#1.load Training and Testing Data
#2.Scale the training data
#3.save scaled data into processed folder
import pandas as pd 
from sklearn.preprocessing import StandardScaler
import pickle
from data_preprocessing import load_and_Split_data
x_train,x_test,y_train,y_test=load_and_Split_data()
scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)
pd.DataFrame(x_train_scaled,columns=x_train.columns).to_csv("C:\\Users\\akhil\\OneDrive\\Documents\\Project\\Insurance_Prediction\\data\\processed\\x_train_scaled.csv",index=False)
pd.DataFrame(x_test_scaled,columns=x_test.columns).to_csv("C:\\Users\\akhil\\OneDrive\\Documents\\Project\\Insurance_Prediction\\data\\processed\\x_test_scaled.csv",index=False)
pd.DataFrame(y_train).to_csv("C:\\Users\\akhil\\OneDrive\\Documents\\Project\\Insurance_Prediction\\data\\processed\\y_train.csv",index=False)
pd.DataFrame(y_test).to_csv("C:\\Users\\akhil\\OneDrive\\Documents\\Project\\Insurance_Prediction\\data\\processed\\y_test.csv",index=False)

with open("C:\\Users\\akhil\\OneDrive\\Documents\\Project\\Insurance_Prediction\\artifacts\\scaler.pkl","wb") as f:
    pickle.dump(scaler,f)