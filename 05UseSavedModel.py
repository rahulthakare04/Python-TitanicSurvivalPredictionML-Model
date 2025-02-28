import joblib
 
model=joblib.load('SurvivalPrediction-model.joblib')

result=model.predict([[3,0,69,0,0,10000]])
print(result)