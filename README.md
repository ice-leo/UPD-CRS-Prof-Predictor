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
* **Flexible Feature Handling**: Accepts schedule entries in military time like `"13:00"` and encodes day, room, semester properly (label or one-hot) to match training data.
* **Model Customization**: Implemented Random Forest and XGBoost (with parameter tuning options) with Random Forest performing better.
* **Streamlit UI**: Interactive web interface to enter class details and get predictions. Another feature is the schedule dashboard, which composes of graphs for easy checking of prof stats.

---

## Installation

```bash
git clone https://github.com/ice-leo/UPD-CRS-Prof-Predictor.git
cd UPD-CRS-Prof-Predictor
```

---

## Usage

### Command Line

Use Python directly:

```python

predicted = predict(
    number=126,
    day="WF",
    room="MB 121",
    semester="2nd sem",
    start_time="13:00",
    end_time="14:30"
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
* **XGBoost**: Tuned using `StratifiedKFold` + `RandomizedSearchCV`, scoring based on accuracy.
* **One-class rule**: For professors with only one class, an exact-match dictionary maps the schedule to the professor directly.

---

## Development

To retrain models:

1. Update `CRS_Math_Profs_Database.ipynb` every year by setting `last_year` to the latest year.
2. Run `CRS_Math_Profs_Predictor_RF.ipynb` or `CRS_Math_Profs_Predictor_XGBoost.ipynb` to preprocess, split, train models, and export the best one.
3. The output includes:

   * `best_rf_model.pkl`
   * `best_xgb_model.pkl`

Contributions welcome! Feel free to file issues or pull requests.
