from pydantic import BaseModel, ConfigDict, Field


class CustomerFeatures(BaseModel):
    """Features client pour la prédiction churn."""

    model_config = ConfigDict(extra="forbid", strict=True)

    gender: str
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(ge=0)

    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str

    Contract: str
    PaperlessBilling: str
    PaymentMethod: str

    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


class PredictionResponse(BaseModel):
    """Réponse de prédiction churn."""

    churn: bool
    probability: float = Field(ge=0.0, le=1.0)
