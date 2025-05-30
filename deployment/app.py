from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from flask import Flask, request, jsonify, render_template
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import pickle
import os
import numpy as np
import pandas as pd

# Flask app for diabetes prediction (Render-deployable)
# Load the trained model
with open("final_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load dataset to calculate model performance
df = pd.read_csv("data/diabetes.csv") 
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_pred = model.predict(X_test)


# Initialize Flask app
app = Flask(__name__, template_folder="templates")

# Home route
@app.route("/")
def home():
    return render_template("home.html")

# JSON Prediction Endpoint
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        input_df = pd.DataFrame([[
            data["Pregnancies"],
            data["Glucose"],
            data["BloodPressure"],
            data["SkinThickness"],
            data["Insulin"],
            data["BMI"],
            data["DiabetesPedigreeFunction"],
            data["Age"]
        ]], columns=["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"])
        prediction = model.predict(input_df)[0]
        return jsonify({"prediction": int(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Generate bar plot
def generate_prediction_plot(prediction):
    fig, ax = plt.subplots()
    bars = ['Non-Diabetic', 'Diabetic']
    values = [1 if i == prediction else 0 for i in range(2)]
    colors = ['green', 'red']

    ax.bar(bars, values, color=colors)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Prediction")

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plot_url = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return plot_url

# Web form route
@app.route("/form", methods=["GET", "POST"])
def form():
    if request.method == "POST":
        try:
            fields = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
            data = {field: float(request.form[field]) for field in fields}

            input_df = pd.DataFrame([list(data.values())], columns=fields)
            prediction = model.predict(input_df)[0]
            plot_url = generate_prediction_plot(prediction)

            return render_template("index.html",
                                   prediction=int(prediction),
                                   plot_url=plot_url)
        except Exception as e:
            return render_template("index.html",
                                   prediction="Error: " + str(e),
                                   plot_url=None)
        
    return render_template("index.html",
                           prediction=None,
                           plot_url=None)


# Run the app
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port, debug=True)