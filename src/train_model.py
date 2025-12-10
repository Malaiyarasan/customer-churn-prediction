import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

DATA_PATH = Path("data/churn_data.csv")

def load_data(path: Path) -> pd.DataFrame:
    """
    Loads churn dataset from a CSV file.
    Expected to contain a binary 'churn' column (0/1 or Yes/No).
    """
    df = pd.read_csv(path)
    return df


def preprocess_split(df: pd.DataFrame):
    # Assume target column is named "churn"
    y = df["churn"]

    # Drop target and any ID column if present
    X = df.drop(columns=["churn", "customer_id"], errors="ignore")

    # Identify categorical and numeric features
    cat_cols = X.select_dtypes(include=["object", "bool"]).columns.tolist()
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # Preprocess pipeline
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return preprocessor, X_train, X_test, y_train, y_test


def build_model(preprocessor):
    model = RandomForestClassifier(
        n_estimators=200, max_depth=None, random_state=42, n_jobs=-1
    )

    clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ])
    return clf


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Please place your churn_data.csv in the data/ folder."
        )

    df = load_data(DATA_PATH)
    preprocessor, X_train, X_test, y_train, y_test = preprocess_split(df)

    clf = build_model(preprocessor)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("Classification report:\n")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()
