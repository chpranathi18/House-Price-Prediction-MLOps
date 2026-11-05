import pandas as pd
import numpy as np
import joblib
import os

from prefect import flow, task

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@task
def load_data():
    df = pd.read_excel(
        "data/House_Price_Prediction_Dataset.xlsx"
    )
    print("Dataset loaded successfully!")
    return df


@task
def train_model(df):
    X = df.drop("Price_in_Lakhs", axis=1)
    y = df["Price_in_Lakhs"]

    numeric_features = X.select_dtypes(include=["number"]).columns
    categorical_features = X.select_dtypes(include=["object"]).columns

    preprocessor = ColumnTransformer([
        ("num", "passthrough", numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=100,
            random_state=42
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model.fit(X_train, y_train)

    print("Model training completed!")

    return model, X_test, y_test


@task
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R2 Score:", r2)

    return mae, rmse, r2


@task
def save_model(model):
    os.makedirs("models", exist_ok=True)

    joblib.dump(model, "models/house_price_model.pkl")

    print("Model saved successfully!")


@flow(name="house-price-training-pipeline")
def house_price_pipeline():
    df = load_data()

    model, X_test, y_test = train_model(df)

    evaluate_model(model, X_test, y_test)

    save_model(model)

    print("House Price Prediction Pipeline Completed!")


if __name__ == "__main__":
    house_price_pipeline()