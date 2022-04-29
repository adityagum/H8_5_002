from distutils.log import debug
from re import template
import flask

import numpy as np
import pickle

model = pickle.load(open('model/predict_model.pkl', 'rb'))

app = flask.Flask(__name__, template_folder='templates')

@app.route('/')
def main():
    return(flask.render_template('index.html'))

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    # int_features = [int(x) for x in flask.request.form.values()]
    int_features = [x for x in flask.request.form.values()]
    final_features = [np.array(int_features)]
    
    prediction = model.predict(final_features)

    #print(prediction)
    #output = {0: 'not placed', 1: 'placed'}

    #return f'##### {prediction}'
    return flask.render_template('index.html', prediction_text='{}'.format(prediction[0]))

if __name__ == "__main__":
    app.run(debug = True)



