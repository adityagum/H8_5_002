from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__) 
model = pickle.load(open('models/modelLR.pkl', 'rb'))
model1 = pickle.load(open('models/modelSvc.pkl', 'rb'))
sc = pickle.load(open('models/scaler.pkl', 'rb'))


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():

    mintemp         = float(request.form['mintemp'])
    rainfall        = float(request.form['rainfall'])
    windgustdir     = float(request.form['windgustdir'])
    windgustspeed   = float(request.form['windgustspeed'])
    windir9         = float(request.form['windir9'])
    windir3         = float(request.form['windir3'])
    windspeed9      = float(request.form['windspeed9'])
    windspeed3      = float(request.form['windspeed3'])
    humidity9       = float(request.form['humidity9'])
    humidity3       = float(request.form['humidity3'])
    raintoday       = float(request.form['raintoday'])
    location        = float(request.form['location'])
    metode          = float(request.form['metode'])


    val = [mintemp, rainfall, windgustdir, windgustspeed, windir9, windir3, windspeed9, windspeed3, humidity9, humidity3]
    val = sc.transform([val])
    val = val.reshape(10,)


    if raintoday == 1:
        val = np.append(val, 1)
    elif raintoday == 0:
        val = np.append(val, 0)
    else:
        print('ERROR!')

    locations = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5,
                6:6, 7:7, 8:8, 9:9, 10:10, 11:11,
                12:12, 13:13, 14:14, 15:15, 16:16, 17:17,
                18:18, 19:19, 20:20, 21:21, 22:22, 23:23,
                24:24, 25:25, 26:26, 27:27, 28:28, 29:29,
                30:30, 31:31, 32:32, 33:33, 34:34, 35:35,
                36:36, 37:37, 38:38, 39:39, 40:40, 41:41, 
                42:42, 43:43, 44:44, 45:45, 46:46}

    for i in range(0,47):
        if locations[location]==i:
            val = np.append(val, 1)
        else:
            val = np.append(val, 0)

    if metode == 1:
        val_predict = model.predict([val])
        return render_template('predict.html', prediction=val_predict)
    elif metode == 2:
        val_predict = model1.predict([val])
        return render_template('predict.html', prediction=val_predict)
    else:
        print('ERROR!')

if __name__ == "__main__":
    app.run(debug=True)