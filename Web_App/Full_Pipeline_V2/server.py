# server.py
from flask import Flask, request, jsonify, render_template
from scraper import vector_generator
from predictor import predict_bus_need  # auto trains on import

app = Flask(__name__, static_folder="static", template_folder="templates")

# Enable CORS if frontend is on a different port during dev
from flask_cors import CORS
CORS(app)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/get_data", methods=["POST"])
def get_data():
    try:
        data = request.get_json(force=True)
        home = data.get("home")
        school = data.get("school")

        if not home or not school:
            return jsonify({"error": "Missing home or school address"}), 400

        final_vec, human_vec = vector_generator(home, school)
        return jsonify(human_vec)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        home = data.get("home")
        school = data.get("school")

        if not home or not school:
            return jsonify({"error": "Missing home or school address"}), 400

        final_vec, human_vec = vector_generator(home, school)
        pred, prob = predict_bus_need(final_vec)

        return jsonify({
            "needs_bus": bool(pred),
            "probability": prob
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Use port 5500 (VSCode Live Server) 
    app.run(debug=True, host="127.0.0.1", port=5500)
