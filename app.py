import os
import logging

from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# ensure logger configuration from project
import src.logger  # noqa: F401  - configures logging

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

    # validate and sanitize input fields
    gender = request.form.get("gender") or ""
    race_ethnicity = request.form.get("ethnicity") or ""
    parental = request.form.get("parental_level_of_education") or ""
    lunch = request.form.get("lunch") or ""
    prep = request.form.get("test_preparation_course") or ""

    # check required text inputs
    for field_name, value in [
        ("gender", gender),
        ("ethnicity", race_ethnicity),
        ("parental_level_of_education", parental),
        ("lunch", lunch),
        ("test_preparation_course", prep),
    ]:
        if not value:
            return f"{field_name} is required", 400

    # safe numeric parsing
    def parse_score(raw):
        if raw is None or raw == "":
            return None
        try:
            return float(raw)
        except (ValueError, TypeError):
            return None

    reading_score = parse_score(request.form.get("reading_score"))
    writing_score = parse_score(request.form.get("writing_score"))
    if reading_score is None or writing_score is None:
        return "Reading and writing scores must be valid numbers", 400

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
    # guard against None/empty results without triggering numpy bool error
    if results is not None and len(results) > 0:
        prediction = results[0]
    else:
        prediction = None
    return render_template("home.html", results=prediction)


if __name__ == "__main__":
    # allow the debug flag to be configured via environment variable
    debug_flag = os.getenv("FLASK_DEBUG", os.getenv("APP_DEBUG", "0"))
    debug = str(debug_flag).lower() in ("1", "true", "yes")
    app.run(host="0.0.0.0", debug=debug)
