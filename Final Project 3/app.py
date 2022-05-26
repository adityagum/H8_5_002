from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__) 
model = pickle.load(open('models/modelRF.pkl', 'rb'))
model1 = pickle.load(open('models/modelETC.pkl', 'rb'))
sc = pickle.load(open('models/scaler.pkl', 'rb'))


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():

    age         = float(request.form['age'])
    anaemia     = float(request.form['anaemia'])
    creatin     = float(request.form['creatin'])
    diabetes    = float(request.form['diabetes'])
    ejectfrac   = float(request.form['ejectfrac'])
    hipertensi  = float(request.form['hipertensi'])
    platelets   = float(request.form['platelets'])
    serum_c     = float(request.form['serum_c'])
    serum_s     = float(request.form['serum_s'])
    sex         = float(request.form['sex'])
    smoking     = float(request.form['smoking'])
    time        = float(request.form['time'])
    metode      = float(request.form['metode'])


    x_input=[[age, anaemia, creatin, diabetes, ejectfrac, hipertensi, platelets, serum_c, serum_s, sex, smoking, time]]
    x_input=sc.fit_transform(x_input)
    x_input=x_input.reshape(12, )
    val_predict = model1.predict([x_input])

    if metode == 1:
        val_predict = model.predict([x_input])
    elif metode == 2:
        val_predict = model1.predict([x_input])
    else:
        print('ERROR!')


    return render_template('predict.html', prediction=val_predict)

if __name__ == "__main__":
    app.run(debug=True)