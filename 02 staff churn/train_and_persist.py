# from IPython.display import display

import numpy as np
import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
)  # , export_text, plot_tree, export_graphviz
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import re
from itertools import combinations
from typing import Iterable, TypeVar, Optional

skPredict = TypeVar(
    "skPredict", LogisticRegression, DecisionTreeClassifier, RandomForestClassifier
)
skDecide = TypeVar(
    "skDecide",
    skPredict,
    LinearRegression,
    DecisionTreeRegressor,
    RandomForestRegressor,
)
skFit = TypeVar("skFit", skPredict, skDecide)


def to_snake_case(name) -> str:
    name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    name = re.sub("([a-z0-9])([A-Z])", r"\1_\2", name)
    return (
        name.replace(" ", "_")
        .replace("__", "_")
        .replace("(", "")
        .replace(")", "")
        .lower()
    )


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.map(to_snake_case)
    return df


def scalar_features(
    df: pd.DataFrame, exclude: Iterable[str] = [], ascending: Optional[bool] = None
) -> Iterable[str]:
    cols = df.select_dtypes(exclude=[object, "category"]).columns
    cols = cols if ascending == None else cols.sort_values(ascending=ascending)
    return [col for col in cols if col not in exclude]


def categorical_features(
    df: pd.DataFrame, exclude: Iterable[str] = [], ascending: Optional[bool] = None
) -> Iterable[str]:
    cols = df.select_dtypes(include=[object, "category"]).columns
    cols = cols if ascending == None else cols.sort_values(ascending=ascending)
    return [col for col in cols if col not in exclude]


def get_lookup_value(
    id: int, df_lookup: pd.DataFrame, col: str = "level", ifNull: str = "Not Answered"
) -> str:
    return (
        f"{ int(id) } { df_lookup.loc[id][col] }" if id in df_lookup.index else ifNull
    )


def validation_testing_training_full_split(
    dataframe: pd.DataFrame,
    seed: int = 42,
    validation: float = 0.2,
    testing: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assert 0 < validation and 0 < testing and 1 > (validation + testing)

    validation_of_full = validation / (1 - testing)
    if validation_of_full == 0:
        validation_of_full = None

    df_full, df_testing = train_test_split(
        dataframe, test_size=testing, random_state=seed, shuffle=True
    )
    df_training, df_validation = train_test_split(
        df_full, test_size=validation_of_full, random_state=seed, shuffle=True
    )

    df_validation = df_validation.reset_index(drop=True)
    df_testing = df_testing.reset_index(drop=True)
    df_training = df_training.reset_index(drop=True)
    df_full = df_full.reset_index(drop=True)

    return df_validation, df_testing, df_training, df_full


def y_split(
    dataframe: pd.DataFrame, yColumn: str, drop: Iterable[str] = []
) -> tuple[pd.DataFrame, pd.Series]:
    columns = set(dataframe.columns)
    assert columns.issuperset([yColumn]), f"{yColumn} not found in dataframe"
    assert columns.issuperset(drop), f"At least one of {drop} not found in dataframe"

    df = dataframe.copy()
    y = df[yColumn]
    for col in [*drop, yColumn]:
        del df[col]

    return df, y


def one_hot_encode(
    df: pd.DataFrame,
    dv: DictVectorizer = DictVectorizer(sparse=False),
    drop: Iterable[str] = [],
    fit: bool = False,
):
    assert set(df.columns).issuperset(
        drop
    ), f"At least one of {drop} is not found in the DataFrame `df`"

    df_encode = df.copy()
    for feature in drop:
        del df_encode[feature]

    data = df_encode.to_dict(orient="records")
    X = dv.fit_transform(data) if fit else dv.transform(data)

    assert len(dv.feature_names_) == X.shape[1]
    return X, dv


def fit(
    model: skFit,
    df: pd.DataFrame,
    y: pd.Series,
    dv: DictVectorizer = DictVectorizer(sparse=False),
    drop: Iterable[str] = [],
) -> tuple[skFit, DictVectorizer]:
    assert df.shape[0] == y.shape[0], "`df` and `y` mismatch"

    X, dv = one_hot_encode(df, dv, drop, fit=True)
    model.fit(X, y)

    return model, dv


def decide(
    model: skDecide, dv: DictVectorizer, df: pd.DataFrame, drop: Iterable[str] = []
):
    X, _ = one_hot_encode(df, dv, drop)
    y_pred = model.predict(X)

    return y_pred


def predict(
    model: skPredict, dv: DictVectorizer, df: pd.DataFrame, drop: Iterable[str] = []
):
    X, _ = one_hot_encode(df, dv, drop)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred


def all_combinations_of(arr):
    return (
        combo
        for choose in range(1 + len(arr))
        for combo in map(list, combinations(arr, choose))
    )


def prepare_training_data() -> pd.DataFrame:
    # Load Datasets

    [dfEducation, dfRating, dfSatisfied] = map(
        standardize_columns,
        [
            pd.read_csv(
                "./data/EducationLevel.csv",
                names=["id", "level"],
                index_col="id",
                skiprows=1,
            ),
            pd.read_csv(
                "./data/RatingLevel.csv",
                names=["id", "level"],
                index_col="id",
                skiprows=1,
            ),
            pd.read_csv(
                "./data/SatisfiedLevel.csv",
                names=["id", "level"],
                index_col="id",
                skiprows=1,
            ),
        ],
    )

    [dfEmployee, dfPerformance] = map(
        standardize_columns,
        [
            pd.read_csv("./data/Employee.csv"),
            pd.read_csv("./data/PerformanceRating.csv"),
        ],
    )

    # Join most-recent Performance data per `employee_id` to Employee data

    dfPerformance.review_date = pd.to_datetime(
        dfPerformance.review_date
    )  # USA dates format is default
    dfPerformance["review_date_ym"] = dfPerformance.review_date.dt.to_period("M")
    dfPerformance["review_date_yq"] = dfPerformance.review_date.dt.to_period("Q")

    dfPerformance_recent = dfPerformance.sort_values(
        by=["employee_id", "review_date"], ascending=[True, False]
    ).drop_duplicates(subset="employee_id", keep="first")

    dfEmployee = pd.merge(
        dfEmployee, dfPerformance_recent, how="left", on="employee_id"
    ).set_index("employee_id")

    # Enrich survey scores with corresponding look-up value

    dfEmployee["education_level"] = dfEmployee.education.apply(
        lambda id: get_lookup_value(id, dfEducation)
    )

    dfEmployee["environment_satisfaction_level"] = (
        dfEmployee.environment_satisfaction.apply(
            lambda id: get_lookup_value(id, dfSatisfied)
        )
    )
    dfEmployee["job_satisfaction_level"] = dfEmployee.job_satisfaction.apply(
        lambda id: get_lookup_value(id, dfSatisfied)
    )
    dfEmployee["relationship_satisfaction_level"] = (
        dfEmployee.relationship_satisfaction.apply(
            lambda id: get_lookup_value(id, dfSatisfied)
        )
    )
    dfEmployee["work_life_balance_level"] = dfEmployee.work_life_balance.apply(
        lambda id: get_lookup_value(id, dfSatisfied)
    )

    dfEmployee["self_rating_level"] = dfEmployee.self_rating.apply(
        lambda id: get_lookup_value(id, dfRating)
    )
    dfEmployee["manager_rating_level"] = dfEmployee.manager_rating.apply(
        lambda id: get_lookup_value(id, dfRating)
    )

    # Correct source data error

    dfEmployee.education_field = dfEmployee.education_field.apply(
        lambda f: "Marketing" if "Marketing" in f else f
    )

    # Engineer features

    dfEmployee.hire_date = pd.to_datetime(dfEmployee.hire_date)
    dfEmployee["hire_date_ym"] = dfEmployee.hire_date.dt.to_period(
        "M"
    )  # f'{ dfEmployee[col].dt.year }-{ dfEmployee[col].dt.month }'
    dfEmployee["hire_date_yq"] = dfEmployee.hire_date.dt.to_period("Q")

    dfEmployee["years_since_hire"] = (
        (dfEmployee.review_date.dt.year - dfEmployee.hire_date.dt.year)
        .fillna(0)
        .apply(lambda y: int(y) if 0 <= y else 0)
    )

    #  Set data types

    for col in ["over_time", "attrition"]:
        dfEmployee[col] = dfEmployee[col].apply(lambda val: 0 if "No" in val else 1)

    for col in [
        "gender",
        "business_travel",
        "department",
        "state",
        "ethnicity",
        "education_field",
        "job_role",
        "marital_status",
        "stock_option_level",
        "education_level",
        "environment_satisfaction_level",
        "job_satisfaction_level",
        "relationship_satisfaction_level",
        "work_life_balance_level",
        "self_rating_level",
        "manager_rating_level",
    ]:
        dfEmployee[col] = dfEmployee[col].astype("category")

    dfEmployee.salary = dfEmployee.salary.astype("UInt32")

    for col in dfEmployee.select_dtypes(include=[int, float]).columns:
        dfEmployee[col] = dfEmployee[col].round().astype("UInt8")

    # Fill missing values

    dfEmployee.fillna(
        {
            "training_opportunities_within_year": 0,
            "training_opportunities_taken": 0,
        },
        inplace=True,
    )

    # drop superfluous features

    del dfEmployee["first_name"]
    del dfEmployee["last_name"]
    del dfEmployee["education"]
    del dfEmployee["hire_date"]
    del dfEmployee["hire_date_ym"]
    del dfEmployee["hire_date_yq"]
    del dfEmployee["performance_id"]
    del dfEmployee["review_date"]
    del dfEmployee["review_date_ym"]
    del dfEmployee["review_date_yq"]

    del dfEmployee["environment_satisfaction"]
    del dfEmployee["job_satisfaction"]
    del dfEmployee["relationship_satisfaction"]
    del dfEmployee["work_life_balance"]

    del dfEmployee["self_rating"]
    del dfEmployee["manager_rating"]

    return dfEmployee


def train_model(
    df: pd.DataFrame,
    target_feature: str,
    seed: int = 1,
    C: float = 1.0,
    drop: Iterable[str] = [],
) -> tuple[LogisticRegression, DictVectorizer]:
    print(
        f"Training model with params: { target_feature = }, { seed = }, { C = }, { drop = }"
    )

    df_val, df_test, df_train, df_full = validation_testing_training_full_split(
        df, seed
    )

    df_val, y_val = y_split(df_val, yColumn=target_feature)
    df_test, y_test = y_split(df_test, yColumn=target_feature)
    df_train, y_train = y_split(df_train, yColumn=target_feature)
    df_full, y_full = y_split(df_full, yColumn=target_feature)

    assert (
        df_val.shape[1] == df_test.shape[1]
        and df_test.shape[1] == df_train.shape[1]
        and df_train.shape[1] == df_full.shape[1]
    )
    assert (
        len(y_val) == df_val.shape[0]
        and len(y_test) == df_test.shape[0]
        and len(y_train) == df_train.shape[0]
        and len(y_full) == df_full.shape[0]
    )

    return fit(
        model=LogisticRegression(
            solver="liblinear", max_iter=1000, random_state=seed, C=C
        ),
        df=df_full,
        y=y_full,
        drop=drop,
    )  # type: ignore


import pickle


def train_and_persist(model_file: str = f"./models/staff-churn.model.bin"):
    target_feature = "attrition"
    best_tuning = {
        "logistic": {"seed": 42, "dropped_features": ["salary", "job_role"], "C": 1.0},
        "random_forest": {
            "seed": 42,
            "dropped_features": ["salary", "ethnicity", "job_role"],
            "depth": 13,
            "estimators": 20,
        },
        "boost": {
            "eta": 0.3,
            "max_depth": 3,
            "min_child_weight": 1,
            "objective": "reg:squarederror",
            "nthread": 2,
            "seed": 42,
            "verbosity": 1,
        },
    }
    dropped_features = best_tuning["logistic"]["dropped_features"]
    C = best_tuning["logistic"]["C"]
    seed = best_tuning["logistic"]["seed"]

    df = prepare_training_data()
    df.info()

    model, dv = train_model(df, target_feature, seed, C, drop=dropped_features)
    print(model, dv)

    with open(model_file, "wb") as f_out:
        pickle.dump((model, dv), file=f_out)

    print(f"Model persisted to {model_file}.")


if __name__ == "__main__":
    train_and_persist()
    print(f"{ __name__ } completed.")
