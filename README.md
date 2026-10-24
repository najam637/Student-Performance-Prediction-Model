# Student Performance Predictor

A beginner-friendly full-stack ML project using scikit-learn, joblib, FastAPI, and HTML/CSS/JavaScript. The website follows the supplied visual reference, with light/dark themes, tabs, grouped slider inputs, and styled prediction panels.

## Models and measured performance

Both models use an 80/20 train-test split (`random_state=42`).

- **Regression:** Linear Regression predicts the numeric `FinalGrade` code. R² = **0.9402** (94.02%); MAE = **0.2358**.
- **Classification:** Random Forest predicts the category mapped from `FinalGrade`. Accuracy = **1.0000** (100.0%).
- Grade mapping used for this demo: `0 = High`, `1 = Good`, `2 = Average`, `3 = Low`. Confirm this mapping with your instructor because the CSV contains numeric codes but no separate official label guide.

These are held-out test results for the supplied dataset, not a guarantee of performance on new data. The high classification score is likely influenced by `ExamScore` being included as an input feature.

## Files

- `student_performance.csv` — dataset
- `train_models.py` — trains both models, prints metrics, and saves model files
- `regression_model.joblib` and `classification_model.joblib` — trained models
- `main.py` — FastAPI app and prediction endpoints
- `static/index.html` — frontend based on the supplied design reference
- `metrics.json` — saved evaluation metrics
- `requirements.txt` — Python dependencies

## Run on Windows

Open a terminal in this folder and run:

```powershell
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

Then visit `http://127.0.0.1:8000`. API docs are at `http://127.0.0.1:8000/docs`.

To retrain the models, run `python train_models.py` from the project folder.

## Required endpoints

- `POST /predict/regression`
- `POST /predict/classification`

The website also uses `GET /dataset/info` to build the slider controls and show metrics, and `GET /dataset/sample` to load a sample student.

## GitHub submission

Create a new GitHub repository and upload these project files. Do not upload `.venv`. Example commands:

```bash
git init
git add .
git commit -m "Update student performance ML app"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
git push -u origin main
```
