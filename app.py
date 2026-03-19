from flask import Flask, request, jsonify
import pickle
import os

app = Flask(__name__)

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return "ML Model is Running 🚀"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = data["features"]
        
        prediction = model.predict([features])
        
        return jsonify({
            "prediction": int(prediction[0])
        })
    
    except Exception as e:
        return jsonify({
            "error": str(e)
        })

if __name__ == "__main__":
    port = 5000
    app.run(host="0.0.0.0", port=port, debug=False)