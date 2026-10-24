# Student Performance Prediction Model

## Project Overview

This project is a full-stack machine learning application that predicts student performance using regression and classification models. It provides a web interface for entering student data and viewing predictions.

## Technologies Used

* Python
* Pandas
* Scikit-learn
* FastAPI
* HTML, CSS, and JavaScript
* Joblib

## Machine Learning Models

* **Regression:** Predicts a numerical student performance value.
* **Classification:** Predicts a student performance category.

The trained models are saved using Joblib and loaded by the backend.

## Project Structure

```text
student_performance_ml_app/
├── main.py
├── train_models.py
├── student_performance.csv
├── regression_model.joblib
├── classification_model.joblib
├── metrics.json
├── requirements.txt
├── README.md
└── static/
    └── index.html
```

## How to Run

1. Install Python.

2. Open PowerShell in the project folder.

3. Create a virtual environment:

   ```powershell
   python -m venv .venv
   ```

4. Activate it:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

5. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

6. Start the FastAPI application using the command appropriate for the project:

   ```powershell
   python -m uvicorn main:app --reload
   ```

7. Open `http://127.0.0.1:8000` in your browser.

8. To test the API, open `http://127.0.0.1:8000/docs`.

## Model Evaluation

Evaluation results are recorded in `metrics.json`. Refer to that file for the actual regression and classification metrics.

## Notes

* Keep the saved model files and `metrics.json` in the locations expected by `main.py`.
* The training script can be used to retrain the models when the dataset is available.
