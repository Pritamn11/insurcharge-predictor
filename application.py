from flask import Flask, request, render_template
import numpy as np

from src.pipeline.predict_pipeline import CustomData, PredictPipeline


application = Flask(__name__)

app = application

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict',methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('pred.html')
    else:
        data = CustomData(
            age= int(request.form.get('age')),
            bmi = float(request.form.get('bmi')),
            children= int(request.form.get('children')),
            sex= request.form.get('sex'),
            smoker= request.form.get('smoker')
        )

    pred_df = data.get_data_as_frame()

    predict_pipeline = PredictPipeline()
    
    result = predict_pipeline.predict(pred_df)

    return render_template('pred.html', result = result[0])


if __name__=="__main__":
    app.run(host="0.0.0.0")