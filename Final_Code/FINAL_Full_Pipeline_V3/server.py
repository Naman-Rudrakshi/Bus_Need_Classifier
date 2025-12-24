# server.py
from flask import Flask, request, jsonify, render_template
from scraper import vector_generator, matrix_generator, df_to_geojson
from predictor import predict_bus_need, predict_bus_need_matrix, rf_models  # auto trains on import
import json
import pandas as pd


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

@app.route("/heatmap_data", methods=["POST"])
def heatmap_data():
    """
    Expects JSON: {"school": "<address>", "radius": <number in miles>}
    Returns GeoJSON FeatureCollection in {"geojson": <geojson_object>}
    """
    try:
        payload = request.get_json(force=True)
        school = payload.get("school")
        radius = payload.get("radius")

        if not school:
            return jsonify({"error": "Missing school address"}), 400
        if radius is None:
            return jsonify({"error": "Missing radius (miles)"}), 400

        # -------------------------------------------------------
        # 1) Generate matrix
        # -------------------------------------------------------
        matrix_df = matrix_generator(school, radius)

        # -------------------------------------------------------
        # 2) Run prediction model
        # -------------------------------------------------------
        matrix_df = predict_bus_need_matrix(matrix_df, rf_models)

        # -------------------------------------------------------
        # 3) Fix rf_prob + create pred = (rf_prob ** 0.35)
        # -------------------------------------------------------

        # force numeric, convert "", "None", etc -> NaN
        matrix_df["rf_prob"] = pd.to_numeric(matrix_df.get("rf_prob"), errors="coerce")

        # clip negative to 0
        matrix_df["rf_prob"] = matrix_df["rf_prob"].clip(lower=0)

        # transformed value for coloring
        matrix_df["pred"] = matrix_df["rf_prob"] ** 0.35

        # ensure geoid is string + zero-padded to 12
        matrix_df["geoid"] = matrix_df["geoid"].astype(str).str.zfill(12)

        # -------------------------------------------------------
        # 4) GeoJSON export (include all columns)
        # -------------------------------------------------------
        geojson_str = df_to_geojson(
            matrix_df,
            geoid_col="geoid",
            lat_col="lat",
            lon_col="lon",
            tiger_year=2023,
            bg_prefix_template="https://www2.census.gov/geo/tiger/TIGER{year}/BG/tl_{year}_{st}_bg.zip",
            props_cols=None,   # all columns included
            verbose=False
        )

        # -------------------------------------------------------
        # 5) Convert GeoJSON string → python dict
        # -------------------------------------------------------
 
        return jsonify({"geojson": json.loads(geojson_str)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Use port 5500 (VSCode Live Server) 
    app.run(debug=True, host="127.0.0.1", port=5500)
