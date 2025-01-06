import pickle
import pandas as pd
import xgboost as xgb

from rich import print

from typing import Optional
from pydantic import BaseModel, Field, field_validator
from fastapi import FastAPI, Query

from train_and_persist import (
    shape_of,
    clean_from_raw,
    feature_engineering,
    preprocessing,
    features_not_compatible_with_modelling,
    features_calculated_as_insignificant,
    features_to_transform,
)


model_pickle = "./models/loan-default.model.bin"
model: xgb.XGBClassifier

with open(model_pickle, "rb") as f_in:
    model = pickle.load(f_in)
    print(f"Model loaded from {model_pickle}.\n", model)


def _predict(model: xgb.XGBClassifier, case: dict) -> int:
    dfCase_processed = (
        pd.DataFrame(data=case, index=[0])
        .pipe(shape_of)
        .assign(
            mobileno_avl_flag=0,
            aadhar_flag=0,
            pan_flag=0,
            voterid_flag=0,
            driving_flag=0,
            passport_flag=0,
            sec_no_of_accts=0,
            sec_active_accts=0,
            sec_overdue_accts=0,
            sec_current_balance=0.0,
            sec_sanctioned_amount=0.0,
            sec_disbursed_amount=0.0,
            sec_instal_amt=0.0,
            loan_default=0,
        )
        .pipe(clean_from_raw)
        .pipe(feature_engineering)
        .drop(features_not_compatible_with_modelling, axis="columns")
        .pipe(preprocessing, features_to_transform)
        .drop("loan_default", axis="columns")
        .drop(features_calculated_as_insignificant, axis="columns")
        .pipe(shape_of)
    )

    # Order the columns as the model expects them
    loan_default: int = model.predict(dfCase_processed[model.feature_names_in_])[0]

    print(f"{case = }")
    print(f"{loan_default = }")

    return loan_default


id_regex = r"^[0-9]+$"
date_regex = r"^\d{2}-\d{2}-\d{2}"  # dd-mm-yy
age_regex = r"^[0-9]+yrs [0-9]+mon"  # 1yrs 11mon


class PredictRequest(BaseModel):
    unique_id: str = Field(pattern=id_regex)
    disbursal_date: str = Field(pattern=date_regex, min_length=8, max_length=8)
    disbursed_amount: float = Field(default=0.0, ge=0.0, examples=[4239.98])
    asset_cost: float = Field(default=0.0, ge=0.0, examples=[4239.98])
    ltv: int = Field(default=0, gt=0, le=100, examples=[49])
    branch_id: str = Field(pattern=id_regex)
    supplier_id: str = Field(pattern=id_regex)
    manufacturer_id: str = Field(pattern=id_regex)
    current_pincode_id: str = Field(pattern=id_regex)
    state_id: str = Field(pattern=id_regex)
    employee_code_id: str = Field(pattern=id_regex)
    perform_cns_score: int = Field(default=0, ge=0, examples=[649])
    pri_no_of_accts: int = Field(default=0, ge=0, examples=[9])
    pri_active_accts: int = Field(default=0, ge=0, examples=[6])
    pri_overdue_accts: int = Field(default=0, ge=0, examples=[4])
    pri_current_balance: float = Field(default=0.0, ge=0.0, examples=[4239.98])
    pri_sanctioned_amount: float = Field(default=0.0, ge=0.0, examples=[4239.98])
    pri_disbursed_amount: float = Field(default=0.0, ge=0.0, examples=[4239.98])
    primary_instal_amt: float = Field(default=0.0, ge=0.0, examples=[4239.98])
    new_accts_in_last_six_months: int = Field(default=0, ge=0, examples=[6])
    delinquent_accts_in_last_six_months: int = Field(default=0, ge=0, examples=[4])
    average_acct_age: str = Field(pattern=age_regex)
    credit_history_length: str = Field(pattern=age_regex)
    no_of_inquiries: int = Field(default=0, ge=0, examples=[4])
    date_of_birth: str = Field(pattern=date_regex, min_length=8, max_length=8)
    employment_type: str = Field(default="Not Answered")
    perform_cns_score_description: str = Field(default="Not Scored")

    @classmethod
    def validate_empty_or_one_of(
        cls, allowed: list[str] = [], val: Optional[str] = None
    ):
        assert val in allowed, f"must be in {allowed}"
        return val

    @field_validator("employment_type")
    def validate_employment_type(cls, val: Optional[str] = None):
        allowed = ["self employed", "salaried"]
        return cls.validate_empty_or_one_of(allowed, val)

    @field_validator("perform_cns_score_description")
    def validate_perform_cns_score_description(cls, val: Optional[str] = None):
        allowed = [
            "very low",
            "not scored",
            "no history",
            "low",
            "medium",
            "high",
            "very high",
        ]
        return cls.validate_empty_or_one_of(allowed, val)


class PredictResponse(BaseModel):
    request: PredictRequest
    loan_default: int
    prediction: bool


api = FastAPI(
    title="Loan Default API",
    description="Predicts whether a loan is likely to default",
)


@api.get("/")
def index() -> dict[str, str]:
    return {"response": "Hello World!"}


@api.get("/weights/")
def weights() -> dict[str, float]:
    return dict(zip(model.feature_names_in_, model.feature_importances_))


@api.post("/predict/")
def predict(request: PredictRequest) -> PredictResponse:
    loan_default = _predict(model, request.model_dump())
    return PredictResponse(
        request=request, loan_default=loan_default, prediction=loan_default == 1
    )


if __name__ == "__main__":
    print(f"{ __name__ } completed.")
