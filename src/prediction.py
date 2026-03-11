import pickle
import numpy as np
class InsurancePremiumPredictor:
    def __init__(self):
        with open("C:\\Users\\akhil\\OneDrive\\Documents\\Project\\Insurance_Prediction\\artifacts\\scaler.pkl","rb") as f:
            self.scaler=pickle.load(f)
        with open("C:\\Users\\akhil\\OneDrive\\Documents\\Project\\Insurance_Prediction\\artifacts\\model.pkl","rb") as f:
            self.model=pickle.load(f)
    def predict(self,Age,Annual_Income_LPA,Policy_Term_Years,Sum_Assured_Lakhs):
        input=np.array([[Age,Annual_Income_LPA,Policy_Term_Years,Sum_Assured_Lakhs]])
        scaled_input=self.scaler.transform(input)
        result=self.model.predict(scaled_input)
        return result[0]