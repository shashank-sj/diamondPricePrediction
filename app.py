from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import CustomData,PredictPipeline
import pandas as pd
import numpy as np 
from src.logger import logging

application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods = ['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    elif request.method == "POST":
        data = CustomData(
        carat=request.form.get('carat'),
        cut=request.form.get('cut'),
        color=request.form.get('color'),
        clarity=request.form.get('clarity'),
        table=request.form.get('table'),
        length=request.form.get('length'),
        width=request.form.get('width'),
        depth=request.form.get('depth')
        )
        pred_df = data.get_data_as_data_frame()
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('home.html', results=results)
    
if __name__=="__main__":
    app.run(host="0.0.0.0")    