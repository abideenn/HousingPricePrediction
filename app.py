import os
from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler



app = Flask(__name__, template_folder='templates')
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/',methods=['GET', 'POST'])
def index():
    return render_template("index.html")


@app.route('/homepage.html',methods=['GET', 'POST'])
def homepage():
    return render_template("homepage.html")


def get_data():
    Latitude = request.form.get('Latitude')
    Longitude = request.form.get('Longitude')
    TotalCharges = request.form.get('Parcelno')
    Landsqfoot = request.form.get('Landsqfoot')
    Lvgarea = request.form.get('Lvgarea')
    SpecFeat = request.form.get('SpecFeat')
    Rail_dist = request.form.get('Rail_dist')
    Ocean_dist = request.form.get('Ocean_dist')
    CNTR_dist = request.form.get('CNTR_dist')
    Subcntr_di = request.form.get('Subcntr_di')
    HWY_dist = request.form.get('HWY_dist')
    Age = request.form.get('Age')
    Monthsold = request.form.get('Monthsold')
    Structure = request.form.get('Structure')

    inputdata=[Latitude,Longitude,Parcelno,Landsqfoot,Lvgarea,SpecFeat,Rail_dist,Ocean_dist,CNTR_dist,Subcntr_di,HWY_dist,Age,Monthsold,Structure]
    array=np.array(inputdata).reshape(1,-1)
    return array


@app.route('/send', methods=['POST'])
def show_data():
    data = get_data()

    logprice = model.predict(data)
    actualprice= np.exp(logprice[0])
    outcome = 'The price of the house is approximately '+str(actualprice)
    return render_template('results.html',result=outcome)

if __name__ == "__main__":
    app.run(debug=True)
