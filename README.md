# UP Diliman CRS Math Professor Predictor

A tool for predicting the course professor of Math classes using class details (course number, day, room, semester, start/end time) powered by Random Forest Classifier (33.26% accuracy).

# Authors
- Mariano, Isaiah John
- Montealto, Meluisa
- Regalario, Jeremiah Daniel

## Overview

UPD CRS Professor Predictor applies classification models (Random Forest & XGBoost) to predict which professor will teach a given Math course based on schedule-based features. It also includes a rule-based shortcut for classes taught by professors who only appear once in the dataset.

---

## Features

* **Hybrid Predictor**: Prioritizes exact schedule–prof mapping for single-class professors, then uses trained ML models for others.
* **Flexible Feature Handling**: Accepts schedule entries like `"1:00PM"` or `"1330"` and encodes day, room, semester properly (label or one-hot) to match training data.
* **Model Customization**: Try both Random Forest and XGBoost (with parameter tuning options).
* **Streamlit UI**: Interactive web interface to enter class details and get predictions.

---

## Installation

```bash
git clone https://github.com/ice-leo/UPD-CRS-Prof-Predictor.git
cd UPD-CRS-Prof-Predictor

# Recommended: create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

pip install -r requirements.txt
```

---

## Usage

### Command Line

Use the Python module directly:

```python
from predictor import predict_professor, load_model

model, encoders, feature_cols = load_model("trained_model.pkl")
predicted = predict_professor(
    number=126,
    day="WF",
    room="MB121",
    semester="2nd sem",
    start_time="1:00PM",
    end_time="2:30PM",
    model=model,
    encoders=encoders,
    feature_columns=feature_cols
)
print("Predicted Professor:", predicted)
```

---

### Streamlit Web App

To launch the interactive web interface:

```bash
streamlit run streamlit_app.py
```

The app provides fields for class number, day, room, semester, and start/end times. It displays the predicted professor.

---

## Model Details

* **Random Forest**: Baseline model with manual hyperparameter tuning.
* **XGBoost**: Tuned using `StratifiedKFold` + `RandomizedSearchCV`, scoring based on macro-F1 to handle class imbalance.
* **One-class rule**: For professors with only one class, an exact-match dictionary maps the schedule to the professor directly.

---

## Development

To retrain models:

1. Update the dataset: `CRS Math Profs (2018-2024).csv`
2. Run `train.py` to preprocess, split, train models, and export the best one.
3. The output includes:

   * `trained_model.pkl`
   * `label_encoders.pkl`
   * `feature_columns.json` for input alignment in the predictor.

Contributions welcome! Feel free to file issues or pull requests.
