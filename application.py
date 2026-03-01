import os
import logging

from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# configure logger from project
import src.logger  # noqa: F401

logger = logging.getLogger(__name__)

application = Flask(__name__)
app = application


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("home.html")

    # validating and sanitizing inputs
    gender = request.form.get("gender") or ""
    race_ethnicity = request.form.get("ethnicity") or ""
    parental = request.form.get("parental_level_of_education") or ""
    lunch = request.form.get("lunch") or ""
    prep = request.form.get("test_preparation_course") or ""

    for fname, val in [
        ("gender", gender),
        ("ethnicity", race_ethnicity),
        ("parental_level_of_education", parental),
        ("lunch", lunch),
        ("test_preparation_course", prep),
    ]:
        if not val:
            return f"{fname} is required", 400

    def parse_score(raw):
        if raw is None or raw == "":
            return None
        try:
            val = float(raw)
        except (ValueError, TypeError):
            return None
        # enforce range 0-100
        if val < 0 or val > 100:
            return None
        return val

    reading_score = parse_score(request.form.get("reading_score"))
    writing_score = parse_score(request.form.get("writing_score"))
    if reading_score is None:
        return "Reading score must be a number between 0 and 100", 400
    if writing_score is None:
        return "Writing score must be a number between 0 and 100", 400

    data = CustomData(
        gender=gender,
        race_ethnicity=race_ethnicity,
        parental_level_of_education=parental,
        lunch=lunch,
        test_preparation_course=prep,
        reading_score=reading_score,
        writing_score=writing_score,
    )

    pred_df = data.get_data_as_data_frame()
    logger.debug("prediction dataframe: %s", pred_df)

    predict_pipeline = PredictPipeline()
    results = predict_pipeline.predict(pred_df)
    # avoid bool(results) for numpy arrays
    if results is not None and len(results) > 0:
        prediction = results[0]
    else:
        prediction = None
    return render_template("home.html", results=prediction)


if __name__ == "__main__":
    debug_flag = os.getenv("FLASK_DEBUG", os.getenv("APP_DEBUG", "0"))
    debug = str(debug_flag).lower() in ("1", "true", "yes")
    app.run(host="0.0.0.0", debug=debug)
