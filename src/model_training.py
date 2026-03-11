#load processed data from processed folder
#2 
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle
x_train=pd.read_csv("C:\\Users\\akhil\\OneDrive\\Documents\\Project\\Insurance_Prediction\\data\\processed\\x_train_scaled.csv")
y_train=pd.read_csv("C:\\Users\\akhil\\OneDrive\\Documents\\Project\\Insurance_Prediction\\data\\processed\\y_train.csv")
x_test=pd.read_csv("C:\\Users\\akhil\\OneDrive\\Documents\\Project\\Insurance_Prediction\\data\\processed\\x_test_scaled.csv")
y_test=pd.read_csv("C:\\Users\\akhil\\OneDrive\\Documents\\Project\\Insurance_Prediction\\data\\processed\\y_test.csv")
print(x_train)
model=LinearRegression()
model.fit(x_train,y_train)
with open("C:\\Users\\akhil\\OneDrive\\Documents\\Project\\Insurance_Prediction\\artifacts\\model.pkl","wb") as f:
    pickle.dump(model,f)