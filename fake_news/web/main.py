import uvicorn
from fastapi import FastAPI
from joblib import load
import preparation
app=FastAPI()
model=load('model.joblib')
@app.get('/')#route par deffault hhtps/rayen/...
def index():
    return {"Welcome to news prediction3 ": "this mini api is so limited to specific news webs "}
@app.post('/predict')#https/rayen/../predict
def predict(titre:str,text : str):
    titre=preparation.clean(titre)
    text=preparation.clean(text)
    prediction=model.predict(preparation.vect(titre, text))  
    if prediction==['1']: return 'news are valide'
    else: return 'news are not valide'

if __name__=='__main___':
    uvicorn.run(app,host='127.0.0.1',port=8000)
    
