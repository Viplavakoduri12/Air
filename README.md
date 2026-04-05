# Viplava - Air Quality Predictor

This project is organized as a normal Python project with separate files for data inspection, preprocessing, model training, evaluation, and the Streamlit application.

Prepared by `Viplava`.

## Python files

- `config.py`: shared file paths and model settings
- `inspect_data.py`: prints dataset preview, dataset info, and missing values
- `data_processing.py`: loads and cleans the dataset
- `model_training.py`: splits the data, trains the XGBoost model, and saves it
- `evaluation.py`: calculates metrics and saves the evaluation plot
- `train_model.py`: runs the full model training pipeline
- `app.py`: Streamlit user interface for AQI prediction
- `main.py`: launcher for the Streamlit app

## Train the model

1. Place `city_day.csv` in the project folder.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Inspect the dataset if needed:

```bash
python inspect_data.py
```

4. Train the model:

```bash
python train_model.py
```

This creates:

- `xgboost_aqi_model.pkl`
- `actual_vs_predicted_aqi.png`

## Run the app

```bash
python main.py
```

or

```bash
run_app.bat
```

## Build the executable

1. Make sure `xgboost_aqi_model.pkl` is in the project folder.
2. In VS Code, select your Python interpreter.
3. Run the task `Build Windows executable`.

The executable will be created in:

```text
dist/AirQualityPredictor/AirQualityPredictor.exe
```
