# Starter pack
from flask import Flask, send_from_directory, render_template
from flask_restful import Api, Resource, reqparse
# from flask_cors import CORS #comment this on deployment
import numpy as np
import joblib
import pickle
from flask import jsonify
from flask import request

"""
app = Flask(__name__, static_url_path='', static_folder='frontend/dist')

Hal itu digunakan untuk menemukan lokasi web yang htmlnya akan dipakai nantinya.
Untuk build itu digunakan saat memakai react
Untuk dist itu digunakan saat memakai vite-react. 

Karena ini memakai vite react + flask, jadinya pakai dist
"""
# app = Flask(__name__, static_url_path='', static_folder='frontend/build')
app = Flask(__name__, static_url_path='', static_folder='frontend/dist')
# CORS(app) #comment this on deployment
api = Api(app)


"""
send_from_directory digunakan saat deploy dengan tujuan 
web yang dideploy itu berhasil menampilkan tampilan web
yang seharusnya

JANGAN LUPA saat push ke heroku, dist-nya jangan di ignore
"""
@app.route("/", defaults={'path':''})
def serve(path):
    return send_from_directory(app.static_folder,'index.html')


if __name__ == '__main__':
    app.run()

"""
Jika tidak ada definisi methods, berarti methods=['GET']
Agar bisa diterima oleh axios, dikirim dengan jsonify()
"""
@app.route("/flask/get")
def users_api():
    message = "Let's Predict This"
    return jsonify(message = message)

# model = joblib.load(open('./model/random_forest.joblib', 'rb'))
model_kmeans = joblib.load(open('./model/model_kmeans.joblib', 'rb'))
scaling =  joblib.load(open('./model/scaling.joblib', 'rb'))

"""
Ini saat form yang telah diisi di POST melalui axios
data dikirim ke route prediction
data form dari axios diambil dengan request.json
yahh sisanya sama seperti projek sebelumnya.
"""
@app.route("/flask/prediction", methods=['POST'])
def jet():
    try:
        dicto = dict(request.json)
        print(dicto.items())
        float_features = [float(value) for key,value in dicto.items()]
    except:
        return jsonify(
            message="Please, just input number!",
        )

    final_features = [np.array(float_features)]
    final_features = scaling.transform(final_features)
    prediction = model_kmeans.predict(final_features)
    print(final_features)
    print(prediction)
    cluster = ['Synchro', 'Synchro2', 'Synchro3', 'Synchro4', 'Synchro5']

    return jsonify(
        message=int(prediction[0]),
    )

@app.errorhandler(404)
def not_found():
    return send_from_directory(app.static_folder,'index.html')

#* nama def di bawah route itu bisa diubah sesuka kita
#* nama url route bisa diubah asalkan entar disesuaikan lagi dengan axios.