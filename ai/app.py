from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

# -----------------------------------------
# CREATE FLASK APPLICATION
# -----------------------------------------

app = Flask(__name__)
CORS(app)


# -----------------------------------------
# LOAD TRAINED ML MODEL
# -----------------------------------------

with open("forest_risk_model.pkl", "rb") as file:
    model = pickle.load(file)


# -----------------------------------------
# RISK LEVELS
# -----------------------------------------

risk_labels = {
    0: "LOW",
    1: "MODERATE",
    2: "HIGH",
    3: "CRITICAL"
}


# -----------------------------------------
# HOME ROUTE
# -----------------------------------------

@app.route("/")
def home():

    return jsonify({
        "project": "Forest Guardian AI",
        "status": "online",
        "message": "AI Risk Prediction API is running"
    })


# -----------------------------------------
# AI FIRE-RISK PREDICTION
# -----------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json()

        # Get input values
        temperature = float(data["temperature"])
        humidity = float(data["humidity"])
        rainfall = float(data["rainfall"])
        dryness = float(data["dryness"])
        previous_fires = float(data["previous_fires"])
        patrol_coverage = float(data["patrol_coverage"])

        # Create feature array
        features = np.array([[
            temperature,
            humidity,
            rainfall,
            dryness,
            previous_fires,
            patrol_coverage
        ]])

        # Make prediction
        prediction = model.predict(features)[0]

        # Get probabilities
        probabilities = model.predict_proba(features)[0]

        # Highest probability
        confidence = max(probabilities) * 100

        risk_level = risk_labels[int(prediction)]

        # Recommendation
        if risk_level == "LOW":

            recommendation = (
                "Continue normal forest patrol operations."
            )

        elif risk_level == "MODERATE":

            recommendation = (
                "Increase monitoring of the forest zone."
            )

        elif risk_level == "HIGH":

            recommendation = (
                "Prioritize this zone for immediate patrol."
            )

        else:

            recommendation = (
                "Immediate patrol verification recommended."
            )

        return jsonify({

            "success": True,

            "risk_level": risk_level,

            "risk_score": round(confidence, 2),

            "recommendation": recommendation,

            "input": {
                "temperature": temperature,
                "humidity": humidity,
                "rainfall": rainfall,
                "dryness": dryness,
                "previous_fires": previous_fires,
                "patrol_coverage": patrol_coverage
            }

        })

    except Exception as e:

        return jsonify({

            "success": False,

            "error": str(e)

        }), 400


# -----------------------------------------
# START SERVER
# -----------------------------------------

if __name__ == "__main__":

    print("")
    print("======================================")
    print("       FOREST GUARDIAN AI")
    print("======================================")
    print("AI Risk Prediction Server")
    print("--------------------------------------")
    print("Server: http://127.0.0.1:5000")
    print("Status: ONLINE")
    print("======================================")
    print("")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )