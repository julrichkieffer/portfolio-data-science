from IPython.display import display
from rich.jupyter import print

import numpy as np
import pandas as pd

# from sklearn.feature_selection import mutual_info_classif, SelectKBest, RFE #, SelectPercentile
# from sklearn.feature_extraction import DictVectorizer
# from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay, roc_auc_score, f1_score #, accuracy_score, roc_curve, root_mean_squared_error, mutual_info_score, auc, plot_confusion_matrix
from sklearn.preprocessing import StandardScaler  # , OneHotEncoder
from sklearn.model_selection import (
    train_test_split,
)  # , KFold, GridSearchCV, RandomizedSearchCV

# from sklearn.linear_model import LogisticRegression, LinearRegression #, SGDClassifier
# from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text #, export_graphviz, plot_tree
# from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor #, GradientBoostingClassifier

import re
from itertools import combinations
from typing import Iterable, TypeVar, Optional, Dict, Callable

from tqdm.auto import tqdm

import xgboost as xgb

import pickle


strip_whitespace = lambda val: val.strip() if type(val) is str else val


def to_snake_case(name) -> str:
    name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    name = re.sub("([a-z0-9])([A-Z])", r"\1_\2", name)
    return (
        name.replace(" ", "_")
        .replace("__", "_")
        .replace(".", "_")
        .replace("(", "")
        .replace(")", "")
        .lower()
    )
    df.columns = df.columns.map(to_snake_case)
    return df


def scalar_features(
    df: pd.DataFrame, exclude: Iterable[str] = [], ascending: Optional[bool] = None
) -> Iterable[str]:
    cols = df.select_dtypes(exclude=[object, "category", bool]).columns
    cols = cols if ascending == None else cols.sort_values(ascending=ascending)
    return [col for col in cols if col not in exclude]


def categorical_features(
    df: pd.DataFrame, exclude: Iterable[str] = [], ascending: Optional[bool] = None
) -> Iterable[str]:
    cols = df.select_dtypes(include=[object, "category"]).columns
    cols = cols if ascending == None else cols.sort_values(ascending=ascending)
    return [col for col in cols if col not in exclude]


def validation_testing_training_full_split(
    dataframe: pd.DataFrame,
    seed: int = 42,
    validation: float = 0.2,
    testing: float = 0.2,
    **kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assert 0 < validation and 0 < testing and 1 > (validation + testing)

    validation_of_full = validation / (1 - testing)
    if validation_of_full == 0:
        validation_of_full = None

    if "stratify" in kwargs:
        stratify = kwargs.pop("stratify")
        stratify_feature = (
            stratify.name
            if isinstance(stratify, pd.Series)
            else stratify if isinstance(stratify, str) else stratify[0]
        )
        assert set(dataframe.columns).issuperset(
            [stratify_feature]
        ), f"{stratify_feature} not found in dataframe"

        df_full, df_testing = train_test_split(
            dataframe,
            test_size=testing,
            random_state=seed,
            stratify=dataframe[stratify_feature],
            **kwargs,
        )
        df_training, df_validation = train_test_split(
            df_full,
            test_size=validation_of_full,
            random_state=seed,
            stratify=df_full[stratify_feature],
            **kwargs,
        )
    else:
        df_full, df_testing = train_test_split(
            dataframe, test_size=testing, random_state=seed, **kwargs
        )
        df_training, df_validation = train_test_split(
            df_full, test_size=validation_of_full, random_state=seed, **kwargs
        )

    df_training = df_training.reset_index(drop=True)
    df_validation = df_validation.reset_index(drop=True)
    df_full = df_full.reset_index(drop=True)
    df_testing = df_testing.reset_index(drop=True)

    return df_training, df_validation, df_full, df_testing


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


def memory_consumption_of(
    df: pd.DataFrame, preamble: str = "memory consumption = ", postamble: str = ""
) -> pd.DataFrame:
    return (
        print(
            f"{ preamble }{ round( df.memory_usage(deep=True).sum() / 1024 / 1024, 2) } MB{ postamble }"
        )
        or df
    )


def shape_of(
    df: pd.DataFrame, preamble: str = "shape = ", postamble: str = ""
) -> pd.DataFrame:
    return print(f"{ preamble }{ df.shape }{ postamble }") or df


def save_to(df: pd.DataFrame, global_key: str = "df") -> pd.DataFrame:
    assert global_key, "a global_key is required"

    globals()[global_key] = df
    print(f"saved to { global_key }")
    return df


def clean_from_raw(dataframe: pd.DataFrame) -> pd.DataFrame:
    def fix_raw_date(series: pd.Series) -> pd.Series:
        """Converts the raw dataset's date values into a Numpy Series of dates"""

        # df['date_col'] =  pd.to_datetime(df['date_col'], format='%d/%m/%Y')
        # df[['start_date', 'end_date']] = df[['start_date', 'end_date']].apply(pd.to_datetime, format="%m/%d/%Y")
        # df['date'] = df['date'].astype('datetime64[ns]') or use datetime64[D] if you want Day precision and not nanoseconds

        # display(series.value_counts(dropna=False).sort_index())
        # ASSUME -> FIX `01-01-00` is 2000, not 1900

        _df = pd.DataFrame(series).join(series.str.split("-", expand=True))
        _df.columns = ["date", "d", "m", "y"]
        _df.loc[_df.y == "00", "c"] = "20"
        _df.fillna({"c": "19"}, inplace=True)
        _df["yyyy"] = _df.c + _df.y

        _df.date = _df.date.str[:6] + _df.yyyy

        return pd.to_datetime(_df.date, dayfirst=True, format="%d-%m-%Y")

    def fix_raw_timedelta(series: pd.Series) -> pd.Series:
        """Converts the raw dataset's time delta values into a Numpy Series of uint8 (months)"""
        # display(series.value_counts(dropna=False).sort_index())

        _df = pd.DataFrame(series).join(series.str.split("yrs ", expand=True))
        _df.columns = ["delta", "yrs", "remainder"]
        _df["months"] = _df.remainder.str.split("mon", expand=True)[0]

        _df = _df.astype(
            {
                "yrs": np.uint8,
                "months": np.uint8,
            }
        )

        return _df.yrs * 12 + _df.months

    def fix_raw_risk(series: pd.Series) -> pd.Series:
        description = series.str.lower()

        no_history = description.str.contains("history") | description.str.contains(
            "enough info"
        )
        very_high = description.str.contains("very high")
        high = description.str.contains("high")
        medium = description.str.contains("medium")
        low = description.str.contains("low")
        very_low = description.str.contains("very low")

        return np.select(
            [very_high, very_low, high, medium, low, no_history],
            ["very high", "very low", "high", "medium", "low", "no history"],
            "not scored",
        )

    return (
        dataframe.rename(
            columns={
                "uniqueid": "unique_id",
                "disbursaldate": "disbursal_date",
            }
        )
        .assign(
            date_of_birth=lambda _: fix_raw_date(_.date_of_birth),
            employment_type=lambda _: _.employment_type.str.lower(),
            disbursal_date=lambda _: fix_raw_date(_.disbursal_date),
            perform_cns_score_description=lambda _: fix_raw_risk(
                _.perform_cns_score_description
            ),
            average_acct_age=lambda _: fix_raw_timedelta(_.average_acct_age),
            credit_history_length=lambda _: fix_raw_timedelta(_.credit_history_length),
        )
        .astype(
            {
                "unique_id": np.uint32,
                "disbursed_amount": np.float32,
                "asset_cost": np.float32,
                "ltv": np.uint8,
                # "branch_id"
                "supplier_id": np.uint16,
                # "manufacturer_id"
                "current_pincode_id": np.uint16,
                # "date_of_birth"
                # "employment_type"
                # "disbursal_date"
                # "state_id"
                "employee_code_id": np.uint16,
                "mobileno_avl_flag": np.uint8,
                "aadhar_flag": np.uint8,
                "pan_flag": np.uint8,
                "voterid_flag": np.uint8,
                "driving_flag": np.uint8,
                "passport_flag": np.uint8,
                "perform_cns_score": np.uint16,
                # "perform_cns_score_description"
                "pri_no_of_accts": np.uint16,
                "pri_active_accts": np.uint16,
                "pri_overdue_accts": np.uint16,
                "pri_current_balance": np.float64,
                "pri_sanctioned_amount": np.float64,
                "pri_disbursed_amount": np.float64,
                "sec_no_of_accts": np.uint16,
                "sec_active_accts": np.uint16,
                "sec_overdue_accts": np.uint16,
                "sec_current_balance": np.float64,
                "sec_sanctioned_amount": np.float64,
                "sec_disbursed_amount": np.float64,
                "primary_instal_amt": np.float64,
                "sec_instal_amt": np.float64,
                "new_accts_in_last_six_months": np.uint8,
                "delinquent_accts_in_last_six_months": np.uint8,
                # "average_acct_age"
                # "credit_history_length"
                "no_of_inquiries": np.uint8,
                "loan_default": np.uint8,
                **{
                    col: "category"
                    for col in [
                        "branch_id",
                        "manufacturer_id",
                        "employment_type",
                        "state_id",
                        "perform_cns_score_description",
                    ]
                },
            }
        )
        .assign(
            perform_cns_score_description=lambda _: _.perform_cns_score_description.cat.reorder_categories(
                [
                    "not scored",
                    "no history",
                    "very low",
                    "low",
                    "medium",
                    "high",
                    "very high",
                ],
                ordered=True,
            )
        )
    )


def feature_engineering(dataframe: pd.DataFrame) -> pd.DataFrame:
    return dataframe.assign(
        date_of_birth_year=lambda _: _.date_of_birth.dt.year,
        date_of_birth_yq=lambda _: _.date_of_birth.dt.to_period("Q"),
        date_of_birth_quarter=lambda _: _.date_of_birth.dt.quarter,
        date_of_birth_ym=lambda _: _.date_of_birth.dt.to_period("M"),
        date_of_birth_month=lambda _: _.date_of_birth.dt.month,
        date_of_birth_dow=lambda _: _.date_of_birth.dt.day_of_week,
        date_of_birth_is_weekend=lambda _: _.date_of_birth.dt.day_of_week.isin([5, 6]),
        disbursal_date_year=lambda _: _.disbursal_date.dt.year,
        disbursal_date_yq=lambda _: _.disbursal_date.dt.to_period("Q"),
        disbursal_date_quarter=lambda _: _.disbursal_date.dt.quarter,
        disbursal_date_ym=lambda _: _.disbursal_date.dt.to_period("M"),
        disbursal_date_month=lambda _: _.disbursal_date.dt.month,
        disbursal_date_dow=lambda _: _.disbursal_date.dt.day_of_week,
        disbursal_date_is_weekend=lambda _: _.disbursal_date.dt.day_of_week.isin(
            [5, 6]
        ),
    )


def preprocessing(
    dataframe: pd.DataFrame,
    features_to_transform: Dict[str, Callable[[pd.Series], pd.DataFrame]] = {},
) -> pd.DataFrame:
    assert len(features_to_transform) >= 0
    assert set(dataframe.columns).issuperset(
        features_to_transform.keys()
    ), f"At least one of {features_to_transform} not found in dataframe"

    df_transformed_features = pd.DataFrame({}, index=dataframe.index)
    for feature, transformer in features_to_transform.items():
        df_ = transformer(dataframe[feature])
        df_transformed_features = df_transformed_features.join(df_)

    df_ = (
        dataframe
        if len(df_transformed_features.columns) == 0
        else (
            dataframe.drop(list(features_to_transform.keys()), axis="columns").join(
                df_transformed_features
            )
        )
    )
    return df_.set_index("unique_id")


def series_scaler(
    series: pd.Series, scaler=StandardScaler(), prefix: str = "", suffix: str = ""
) -> pd.DataFrame:
    assert isinstance(series, pd.Series), "series is required"

    try:
        X = scaler.fit_transform(pd.DataFrame(series))
        return pd.DataFrame(
            X, index=series.index, columns=[f"{prefix}{series.name}{suffix}"]
        )
    except Exception as xp:
        print(f"{ series_scaler.__name__ }: { xp = }")
        return pd.DataFrame(series, index=series.index)


def top_n_encoder(series: pd.Series, top_n: int = 10) -> pd.DataFrame:
    assert isinstance(series, pd.Series), "series is required"
    assert top_n >= 1, "top_n must be greater than 0"

    try:
        df_X = pd.DataFrame({}, index=series.index)

        top_n_series = series.value_counts(ascending=False).head(top_n)
        top_n_values = list(top_n_series.index.values)

        for value in top_n_values:
            df_X[f"{series.name}={value}"] = np.where(series == value, 1, 0).astype(
                "uint8"
            )

        print(
            f"{ top_n_encoder.__name__ }: {top_n_series.sum() / series.value_counts( dropna=False ).sum():.2%} of { series.name } values: { top_n_values }"
        )
        return df_X
    except Exception as xp:
        print(f"{ top_n_encoder.__name__ }: { xp = }")
        return pd.DataFrame(series, index=series.index)


def main():
    # from scipy.io import arff   # FAILED as "NotImplementedError: String attributes not supported yet"
    import arff

    print("Loading raw data...")
    raw = arff.load(open("./data/LT-Vehicle-Loan-Default-Prediction.arff", "r"))
    print("Raw data loaded.")

    # print(raw.keys())
    # print(raw['attributes'])

    columns = [to_snake_case(name) for name, type in raw["attributes"]]
    # print(f"{columns = }")

    print("Pre-processing data...")
    dfLoans = (
        pd.DataFrame(np.array(raw["data"]), columns=columns)
        # .pipe(save_to, "df_raw")
        .pipe(shape_of)
        .pipe(memory_consumption_of)
        .pipe(clean_from_raw)
        # .pipe(shape_of)
        # .pipe(memory_consumption_of)
        # .pipe(save_to, "df_cleaned")
        .pipe(feature_engineering)
        # .pipe(shape_of)
        # .pipe(memory_consumption_of)
    )

    del arff, raw, columns
    # del df_raw, df_cleaned

    # dfLoans.head().T
    # dfLoans.describe(include='all').T
    # dfLoans.info()

    seed = 42
    target_feature = "loan_default"

    features_not_compatible_with_modelling = [
        "date_of_birth",
        "date_of_birth_yq",
        "date_of_birth_ym",
        "disbursal_date",
        "disbursal_date_yq",
        "disbursal_date_ym",
    ]

    features_to_transform = {
        "employment_type": lambda _: top_n_encoder(_, 2),
        "perform_cns_score_description": lambda _: top_n_encoder(_),
    }

    print(f"{target_feature} == 1:  {dfLoans[target_feature].mean():.1%}")

    features_calculated_as_insignificant = [
        "mobileno_avl_flag",
        "aadhar_flag",
        "pan_flag",
        "voterid_flag",
        "driving_flag",
        "passport_flag",
        # 'pri_active_accts',
        "sec_no_of_accts",
        "sec_active_accts",
        "sec_overdue_accts",
        "sec_instal_amt",
        "date_of_birth_quarter",
        "date_of_birth_is_weekend",
        "disbursal_date_year",
        "disbursal_date_is_weekend",
        "sec_current_balance",
        "sec_sanctioned_amount",
        "sec_disbursed_amount",
    ]

    dfLoans_processed = (
        dfLoans
        # .pipe(shape_of)
        # .pipe(memory_consumption_of)
        .drop(features_not_compatible_with_modelling, axis="columns")
        .pipe(preprocessing, features_to_transform)
        .drop(features_calculated_as_insignificant, axis="columns")
        .pipe(shape_of)
        .pipe(memory_consumption_of)
    )
    print("Data prepared for training.")

    # display(dfLoans_processed.describe(include="all").T)

    print("Training model...")

    _, _, df_full, df_test = validation_testing_training_full_split(
        dfLoans_processed, seed=seed, shuffle=True, stratify=target_feature
    )
    df_test, y_test = y_split(df_test, yColumn=target_feature)
    df_full, y_full = y_split(df_full, yColumn=target_feature)

    best_tuning = {  # tuning results
        "xgboost": {
            "eta": 0.25,
            "scale_pos_weight": 2.0,
            "max_depth": 5.0,
            "n_estimators": 69.0,
        }
    }

    eta, scale_pos_weight, max_depth, n_estimators = best_tuning["xgboost"].values()

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        learning_rate=eta,
        scale_pos_weight=scale_pos_weight,
        max_depth=int(max_depth),
        n_estimators=int(n_estimators),
        colsample_bytree=0.8,
        enable_categorical=True,
        subsample=0.8,
        random_state=seed,
        early_stopping_rounds=50,
        n_jobs=-1,
    )
    model.fit(df_full, y_full, eval_set=[(df_test, y_test)], verbose=False)

    print(f"training data score: { model.score(df_full, y_full):.2%}")
    print(f"    test data score: { model.score(df_test, y_test):.2%}")
    print("XGBoost model trained.")

    print("Persisting model...")
    model_pickle = "./models/xgboost.model"
    with open(model_pickle, "wb") as f_out:
        pickle.dump(model, f_out)
    print(f"XGBoost model saved to {model_pickle}")


if __name__ == "__main__":
    main()
