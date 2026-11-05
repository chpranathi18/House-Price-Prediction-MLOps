# House Price Prediction using MLOps

## Project Overview
This project predicts house prices using Machine Learning and demonstrates an end-to-end MLOps workflow.

It uses a synthetic house price dataset and integrates tools for model training, experiment tracking, workflow automation, API deployment, monitoring, and containerization.

## Technologies Used
- Python
- Pandas and NumPy
- Scikit-learn
- MLflow – Experiment Tracking
- Prefect – Workflow Orchestration
- FastAPI – API Development
- Prometheus – Monitoring
- Evidently AI – Data Drift Detection
- Docker – Containerization
- Git and GitHub – Version Control

## Project Workflow
1. Load and preprocess the house price dataset.
2. Train a Machine Learning model.
3. Evaluate model performance.
4. Track experiments using MLflow.
5. Automate the workflow using Prefect.
6. Deploy the model using FastAPI.
7. Monitor API performance using Prometheus.
8. Detect data drift using Evidently.
9. Containerize the application using Docker.

## Model Performance

| Metric | Result |
|---|---:|
| MAE | 7.63 |
| RMSE | 9.31 |
| R² Score | 0.897 |

## API Endpoints
- `/` – Home
- `/health` – Health Check
- `/predict` – House Price Prediction
- `/metrics` – Prometheus Metrics

## Docker
The application is containerized using Docker.

```bash
docker build -t house-price-api .
docker run -d -p 8000:8000 --name house-price-container house-price-api
```

Open the API documentation:

`http://127.0.0.1:8000/docs`

## Project Structure

```text
House-Price-Prediction-MLOps/
│
├── app.py
├── house_prefect_flow.py
├── Dockerfile
├── requirements.txt
├── .dockerignore
├── .gitignore
├── models/
├── House_Price_Drift_Detection
├── house_price_mlop_practical.ipynb
└── Untitled.ipynb
```

## Conclusion
This project demonstrates how Machine Learning models can be developed, tracked, automated, deployed, monitored, and maintained using MLOps tools.
