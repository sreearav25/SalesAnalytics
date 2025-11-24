"""Sales, salary and profit analytics.

Author: Aravind
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


def load_financial_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    if "profit" not in df.columns:
        df["profit"] = df["revenue"] - df["cogs"] - df["salary_expense"] - df["other_expense"]
    return df


def basic_kpis(df: pd.DataFrame) -> dict:
    return {
        "total_revenue": float(df["revenue"].sum()),
        "total_profit": float(df["profit"].sum()),
        "avg_profit_margin": float((df["profit"].sum() / df["revenue"].sum())
                                   if df["revenue"].sum() else 0.0),
        "avg_salary_expense": float(df["salary_expense"].mean()),
    }


def train_sales_model(df: pd.DataFrame):
    """Train a simple model to predict revenue based on calendar features and salary.

    Features:
    - month
    - year
    - salary_expense
    - other_expense
    """
    df = df.copy()
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year

    feature_cols = ["month", "year", "salary_expense", "other_expense"]
    X = df[feature_cols]
    y = df["revenue"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)

    return model, mae, X_test, y_test, y_pred


def predict_future_revenue(model, future_df: pd.DataFrame) -> pd.Series:
    """Given a future dataframe with same feature columns, return predicted revenue."""
    feature_cols = ["month", "year", "salary_expense", "other_expense"]
    return pd.Series(model.predict(future_df[feature_cols]), index=future_df.index)
