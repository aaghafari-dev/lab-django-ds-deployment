import os
import joblib
import numpy as np
from django.conf import settings
from django.shortcuts import render

# Load the trained model once at import time
MODEL = joblib.load(settings.MODEL_PATH)

# Iris metadata (must match training order)
FEATURE_NAMES = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]
CLASS_NAMES = ["setosa", "versicolor", "virginica"]


def home(request):
    """Display the input form."""
    return render(request, "predictor/home.html", {"feature_names": FEATURE_NAMES})


def predict(request):
    """Handle form submission and return a prediction."""
    if request.method != "POST":
        return render(request, "predictor/home.html", {"feature_names": FEATURE_NAMES})

    errors = []
    values = []

    for name in FEATURE_NAMES:
        raw = request.POST.get(name, "").strip()
        if raw == "":
            errors.append(f"Missing value for {name}.")
            continue
        try:
            values.append(float(raw))
        except ValueError:
            errors.append(f"Invalid number for {name}: '{raw}'.")

    if errors:
        return render(
            request,
            "predictor/home.html",
            {"feature_names": FEATURE_NAMES, "errors": errors, "posted": request.POST},
        )

    X = np.array(values).reshape(1, -1)
    pred_idx = int(MODEL.predict(X)[0])
    proba = MODEL.predict_proba(X)[0]
    pred_label = CLASS_NAMES[pred_idx]

    context = {
        "feature_names": FEATURE_NAMES,
        "inputs": dict(zip(FEATURE_NAMES, values)),
        "prediction": pred_label,
        "probabilities": [
            {"class": CLASS_NAMES[i], "prob": round(float(proba[i]) * 100, 2)}
            for i in range(len(CLASS_NAMES))
        ],
    }
    return render(request, "predictor/result.html", context)
