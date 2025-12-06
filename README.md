# VitalSense: AI-Powered Health Monitoring System

VitalSense is a comprehensive health monitoring system designed to predict and monitor the risk of Diabetes and Heart Disease using Machine Learning.

## Features
- **AI-Powered Prediction**: Uses XGBoost models to predict disease risk based on patient vitals.
- **Live Monitoring Dashboard**: A Streamlit-based dashboard for real-time monitoring and simulation.
- **Medical-Aware Preprocessing**: Handles missing data and outliers using domain-specific medical rules.
- **Privacy Focused**: Anonymized data processing and secure configuration.

## Project Structure
- `src/`: Source code for training, preprocessing, and monitoring.
    - `train.py`: Trains the XGBoost models.
    - `live_monitor.py`: Runs the interactive dashboard.
    - `preprocess.py`: Cleans and prepares the dataset.
- `data/`: Directory for datasets (excluded from git).
- `model/`: Directory for saved ML models.

## Usage

### 1. Preprocess Data
Prepare the dataset (if you have raw data):
```bash
python src/preprocess.py
```

### 2. Train Models
Train the Diabetes and Heart Disease models:
```bash
python src/train.py
```

### 3. Run Dashboard
Launch the live monitoring dashboard:
```bash
streamlit run src/live_monitor.py
```

## License
[MIT License](LICENSE)
