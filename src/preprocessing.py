import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

TRAIN_PATH = Path("data") / "hotel_bookings.csv"


def load_data(train_path=TRAIN_PATH):
    """
    Load dataset and separate features from target.
    """

    df = pd.read_csv(train_path)

    # Remove unnecessary columns
    df.drop(
        columns=[
            "company",
            "reservation_status",
            "reservation_status_date",
        ],
        inplace=True,
        errors="ignore",
    )

    X = df.drop("is_canceled", axis=1)
    y = df["is_canceled"]

    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into train and test sets.
    """

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def handle_missing_values(X_train, X_test, strategy="median"):
    """
    Handle missing values using statistics learned
    from the training data only.
    """

    X_train = X_train.copy()
    X_test = X_test.copy()

    # agent -> fill with zero
    for df in (X_train, X_test):
        if "agent" in df.columns:
            df["agent"] = df["agent"].fillna(0)

    numeric_cols = X_train.select_dtypes(include="number").columns
    categorical_cols = X_train.select_dtypes(exclude="number").columns

    numeric_imputer = SimpleImputer(strategy=strategy)
    categorical_imputer = SimpleImputer(strategy="most_frequent")

    # Train
    X_train[numeric_cols] = numeric_imputer.fit_transform(
        X_train[numeric_cols]
    )

    X_train[categorical_cols] = categorical_imputer.fit_transform(
        X_train[categorical_cols]
    )

    # Test
    X_test[numeric_cols] = numeric_imputer.transform(
        X_test[numeric_cols]
    )

    X_test[categorical_cols] = categorical_imputer.transform(
        X_test[categorical_cols]
    )

    return X_train, X_test


def encode_categorical_features(X_train, X_test):
    """
    One-hot encode categorical features.
    """

    categorical_cols = [
        "hotel",
        "meal",
        "market_segment",
        "distribution_channel",
        "deposit_type",
        "customer_type",
        "reserved_room_type",
        "assigned_room_type",
        "arrival_date_month",
        "country",
    ]

    X_train = pd.get_dummies(
        X_train,
        columns=categorical_cols,
        drop_first=True,
        dtype=int,
    )

    X_test = pd.get_dummies(
        X_test,
        columns=categorical_cols,
        drop_first=True,
        dtype=int,
    )

    # Make train/test have identical columns
    X_train, X_test = X_train.align(
        X_test,
        join="left",
        axis=1,
        fill_value=0,
    )

    return X_train, X_test


def scale_features(X_train, X_test):
    """
    Apply Standard Scaling.
    """

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)

    X_test = scaler.transform(X_test)

    return X_train, X_test