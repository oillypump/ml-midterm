# predict.py
import pickle
from flask import Flask, request, jsonify

MODEL_FILE = "model.bin"

with open(MODEL_FILE, "rb") as f_in:
    dv, model = pickle.load(f_in)

app = Flask("hotel-cancellation")


@app.route("/predict", methods=["POST"])
def predict():
    booking = request.get_json()

    # res_month harus string, karena saat training di-one-hot
    booking["res_month"] = str(booking["res_month"])

    X = dv.transform([booking])
    prob = float(model.predict_proba(X)[0, 1])

    return jsonify(
        {
            "cancel_probability": prob,
            "cancel": prob >= 0.5,
        }
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
