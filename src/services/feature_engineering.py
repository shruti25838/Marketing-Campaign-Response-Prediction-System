import pandas as pd
import numpy as np


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add demographic, behavioral, and interaction features for campaign response prediction."""
    df = df.copy()

    # --- Tenure / signup (if available) ---
    if "signup_date" in df.columns:
        df["signup_date"] = pd.to_datetime(df["signup_date"], errors="coerce")
        df["customer_tenure_days"] = (
            pd.Timestamp.utcnow().normalize() - df["signup_date"]
        ).dt.days

    # --- Aggregated behavioral (by customer_id if present) ---
    if {"purchase_amount", "customer_id"}.issubset(df.columns):
        df["total_previous_purchases"] = df.groupby("customer_id")["purchase_amount"].transform(
            "sum"
        )
    if {"response", "customer_id"}.issubset(df.columns):
        df["past_response_rate"] = df.groupby("customer_id")["response"].transform(
            "mean"
        )

    # --- Channel / segment interaction ---
    if {"campaign_channel", "customer_segment"}.issubset(df.columns):
        df["channel_segment"] = (
            df["campaign_channel"].astype(str) + "_" + df["customer_segment"].astype(str)
        )

    # --- Numeric ratios ---
    if {"balance", "age"}.issubset(df.columns):
        df["balance_per_age"] = df["balance"] / df["age"].replace(0, np.nan)
    if {"campaign", "duration"}.issubset(df.columns):
        safe_campaign = df["campaign"].replace(0, np.nan)
        df["duration_per_contact"] = df["duration"] / safe_campaign

    # --- Previous contact (pdays) ---
    if "pdays" in df.columns:
        df["has_previous_contact"] = (df["pdays"] >= 0).astype(int)
        df["days_since_last_contact"] = df["pdays"].replace(-1, np.nan)

    # --- Interaction features (demographic × demographic) ---
    if {"job", "marital"}.issubset(df.columns):
        df["job_marital"] = df["job"].astype(str) + "_" + df["marital"].astype(str)
    if {"job", "education"}.issubset(df.columns):
        df["job_education"] = df["job"].astype(str) + "_" + df["education"].astype(str)
    if {"contact", "month"}.issubset(df.columns):
        df["contact_month"] = df["contact"].astype(str) + "_" + df["month"].astype(str)

    # --- Age segments (behavioral / demographic bucket) ---
    if "age" in df.columns:
        df["age_group"] = pd.cut(
            df["age"],
            bins=[0, 25, 35, 45, 55, 65, 120],
            labels=["18-25", "26-35", "36-45", "46-55", "56-65", "65+"],
        ).astype(str)

    # --- Past outcome (poutcome: success/failure/unknown) ---
    if "poutcome" in df.columns:
        df["previous_success"] = (df["poutcome"].astype(str).str.lower() == "success").astype(int)
        df["previous_failure"] = (df["poutcome"].astype(str).str.lower() == "failure").astype(int)

    # --- Engagement / contact intensity (aggregated behavioral metrics) ---
    if "previous" in df.columns:
        df["total_contacts_ever"] = df["previous"]  # often same as previous
    if {"campaign", "previous"}.issubset(df.columns):
        df["total_campaign_contacts"] = df["campaign"] + df["previous"].fillna(0)
    if "duration" in df.columns:
        df["log_duration"] = np.log1p(df["duration"].clip(lower=0))
    if "balance" in df.columns:
        df["log_balance"] = np.log1p(df["balance"].clip(lower=0))

    return df
