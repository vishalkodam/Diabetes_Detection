import pandas as pd
from flask import Flask, render_template, flash, request
from get_model import *
import pickle

# initializing the model
MODEL_PATH = r'C:\Users\visha\OneDrive\Desktop\diabetes-detection-master\diabetes-detection-master\TrainedModel.dat'
model = pickle.load(open(MODEL_PATH, "rb"))

# App config.
DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'


# Define home route
@app.route("/")
def index():
    return render_template("index.html")


# Define diagnosis route
@app.route("/diagnosis", methods=['POST'])
def diagnosis():
    pregnant = request.form['pregnant']
    glucose = request.form['glucose']
    bp = request.form['bp']
    name = request.form['name']
    insulin = request.form['insulin']
    bmi = request.form['bmi']
    pedigree = request.form['pedigree']
    age = request.form['age']


    data = {
        'Pregnancies': pregnant,
        'Glucose': glucose,
        'BloodPressure': bp,
        'SkinThickness': name,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': pedigree,
        'Age': age,
    }

    df = pd.DataFrame(data, index=[0])
    print(df)

    # Predict on the given parameters
    prediction = model.predict(df)
    print(prediction)
    # Route for result
    if prediction[0] == 1:
        return render_template("positive.html", result="true")
    elif prediction[0] == 0:
        return render_template("negetive.html", result="true")

if __name__ == "__main__":
    app.run()
