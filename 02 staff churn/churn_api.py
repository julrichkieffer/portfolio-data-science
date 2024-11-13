import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from typing import Optional
from pydantic import BaseModel, Field, field_validator
from fastapi import FastAPI, Query


dv: DictVectorizer
model: LogisticRegression

model_file = "./models/staff-churn.model.bin"
with open(model_file, "rb") as f_in:
    # pickle.dump((model, dv), file=f_out)
    (model, dv) = pickle.load(f_in)

    print(f"Model loaded from {model_file}.\n", model, dv)


def _predict(case) -> float:
    X = dv.transform([case])  # type: ignore
    probability: float = model.predict_proba(X)[0, 1]

    print(f"{case        = }")
    print(f"{probability = }")

    return probability


def _decide(probability: float, threshold: float) -> bool:
    decision = threshold < probability

    print(f"{threshold   = }")
    print(f"{decision    = }")

    return decision


class PredictRequest(BaseModel):
    age: Optional[int] = Field(default=0, gt=0, lt=120, examples=[49])
    business_travel: Optional[str] = None
    department: Optional[str] = None
    distance_from_home_km: Optional[float] = Field(default=0, gt=0, examples=[49])
    education_field: Optional[str] = None
    education_level: Optional[int | str] = Field(default=0, gt=0, lt=6, examples=[4])
    environment_satisfaction_level: Optional[int | str] = Field(
        default="Not Answered", gt=0, lt=6, examples=[4]
    )
    ethnicity: Optional[str] = None
    gender: Optional[str] = None
    job_satisfaction_level: Optional[int | str] = Field(
        default="Not Answered", gt=0, lt=6, examples=[4]
    )
    manager_rating_level: Optional[int | str] = Field(
        default="Not Answered", gt=0, lt=6, examples=[4]
    )
    marital_status: Optional[str] = None
    over_time: Optional[int] = Field(default=0, gt=-1, lt=2, examples=[1])

    @classmethod
    def validate_empty_or_one_of(
        cls, allowed: list[str] = [], val: Optional[str] = None
    ):
        assert val in allowed, f"must be in {allowed}"
        return val

    @classmethod
    def validate_rating_level(cls, val: Optional[int | str] = None):
        allowed = [
            "Not Answered",
            "1 Unacceptable",
            "2 Needs Improvement",
            "3 Meets Expectation",
            "4 Exceeds Expectation ",
            "5 Above and Beyond",
        ]
        if val is None:
            return val
        elif isinstance(val, str):
            assert val in allowed, f"must be in {allowed}"
        else:
            return allowed[val]

        return None

    @classmethod
    def validate_satisfaction_level(cls, val: Optional[int | str] = None):
        allowed = [
            "Not Answered",
            "1 Very Dissatisfied",
            "2 Dissatisfied",
            "3 Neutral",
            "4 Satisfied ",
            "5 Very Satisfied",
        ]
        if val is None:
            return val
        elif isinstance(val, str):
            assert val in allowed, f"must be in {allowed}"
        else:
            return allowed[val]

        return None

    @field_validator("business_travel")
    def validate_business_travel(cls, val: Optional[str] = None):
        allowed = ["Frequent Traveller", "No Travel ", "No Travel", "Some Travel"]
        return cls.validate_empty_or_one_of(allowed, val)

    @field_validator("department")
    def validate_department(cls, val: Optional[str] = None):
        allowed = ["Human Resources", "Sales", "Technology"]
        return cls.validate_empty_or_one_of(allowed, val)

    @field_validator("education_field")
    def validate_education_field(cls, val: Optional[str] = None):
        allowed = [
            "Business Studies",
            "Computer Science",
            "Economics",
            "Human Resources",
            "Information Systems",
            "Marketing",
            "Other",
            "Technical Degree",
        ]
        return cls.validate_empty_or_one_of(allowed, val)

    @field_validator("education_level")
    def validate_education_level(cls, val: Optional[int | str] = None):
        allowed = [
            "",
            "1 No Formal Qualifications",
            "2 High School ",
            "3 Bachelors ",
            "4 Masters ",
            "5 Doctorate",
        ]
        if val is None:
            return val
        elif isinstance(val, str):
            assert val in allowed, f"must be in {allowed}"
        else:
            return allowed[val]

        return None

    @field_validator("environment_satisfaction_level")
    def validate_environment_satisfaction_level(cls, val: Optional[int | str] = None):
        return cls.validate_satisfaction_level(val)

    @field_validator("ethnicity")
    def validate_ethnicity(cls, val: Optional[str] = None):
        allowed = [
            "American Indian or Alaska Native",
            "Asian or Asian American",
            "Black or African American",
            "Mixed or multiple ethnic groups",
            "Native Hawaiian ",
            "Other ",
            "White",
        ]
        return cls.validate_empty_or_one_of(allowed, val)

    @field_validator("gender")
    def validate_gender(cls, val: Optional[str] = None):
        allowed = ["Female", "Male", "Non-Binary", "Prefer Not To Say"]
        return cls.validate_empty_or_one_of(allowed, val)

    @field_validator("job_satisfaction_level")
    def validate_job_satisfaction_level(cls, val: Optional[int | str] = None):
        return cls.validate_satisfaction_level(val)

    @field_validator("manager_rating_level")
    def validate_manager_rating_level(cls, val: Optional[int | str] = None):
        return cls.validate_rating_level(val)

    @field_validator("marital_status")
    def validate_marital_status(cls, val: Optional[str] = None):
        allowed = ["Divorced", "Married", "Single"]
        return cls.validate_empty_or_one_of(allowed, val)

    # @field_validator("department")
    # def validate_department(cls, val: Optional[str] = None):
    #     allowed = []
    #     return cls.validate_empty_or_one_of(allowed, val)


class PredictResponse(BaseModel):
    request: PredictRequest
    probability: float


class DecideResponse(PredictResponse):
    threshold: float
    decision: bool


api = FastAPI(
    title="Staff Churn API",
    description="Predicts whether a member of staff is likely to churn",
)


@api.get("/")
def index() -> dict[str, str]:
    return {"response": "Hello World!"}


@api.get("/weights/")
def weights() -> dict[str, float]:
    return dict(zip(dv.feature_names_, model.coef_[0]))


@api.post("/predict/")
def predict(request: PredictRequest) -> PredictResponse:
    probability = _predict(request.model_dump())

    return PredictResponse(request=request, probability=probability)


@api.post("/decide")
def decide(
    request: PredictRequest, threshold: float = Query(default=0.5, gt=0.0, lt=1.0)
) -> DecideResponse:
    probability = _predict(request.model_dump())
    decision = _decide(probability, threshold)

    return DecideResponse(
        request=request,
        probability=probability,
        threshold=threshold,
        decision=decision,
    )


if __name__ == "__main__":
    print(f"{ __name__ } completed.")
