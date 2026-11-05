# train.py
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score
import xgboost as xgb

DATA_PATH = "data/booking.csv"
OUTPUT_FILE = "model.bin"

cat_cols = ["type_of_meal", "room_type", "market_segment_type", "res_month"]
num_cols = [
    "number_of_adults",
    "number_of_children",
    "number_of_weekend_nights",
    "number_of_week_nights",
    "car_parking_space",
    "lead_time",
    "repeated",
    "p_c",
    "p_not_c",
    "average_price",
    "special_requests",
]
features = cat_cols + num_cols

params = dict(
    n_estimators=100,
    max_depth=10,
    learning_rate=0.2,
    min_child_weight=1,
    random_state=1,
)


def load_data(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower().str.replace(" ", "_").str.replace("-", "_")

    cat_cols_raw = df.select_dtypes(
        include=["object", "category", "str"]
    ).columns.tolist()
    for c in cat_cols_raw:
        df[c] = df[c].str.lower().str.replace(" ", "_")

    df["is_canceled"] = (df.booking_status == "canceled").astype(int)
    df = df.drop(columns="booking_status")

    # 2018-2-29 tidak valid -> 2018-02-28
    df.loc[df["date_of_reservation"] == "2018-2-29", "date_of_reservation"] = (
        "2/28/2018"
    )
    df["date_of_reservation"] = pd.to_datetime(df["date_of_reservation"])
    df["res_month"] = df["date_of_reservation"].dt.month.astype("str")
    return df


def train(df_train, y_train):
    dv = DictVectorizer(sparse=False)
    X = dv.fit_transform(df_train[features].to_dict(orient="records"))
    model = xgb.XGBClassifier(**params)
    model.fit(X, y_train)
    return dv, model


def predict(df, dv, model):
    X = dv.transform(df[features].to_dict(orient="records"))
    return model.predict_proba(X)[:, 1]


df = load_data(DATA_PATH)

df_full_train, df_test = train_test_split(
    df, test_size=0.2, random_state=1, stratify=df["is_canceled"]
)

# K-Fold CV (5 fold) di df_full_train
kfold = KFold(n_splits=5, shuffle=True, random_state=1)
scores = []
for i, (tr_idx, va_idx) in enumerate(kfold.split(df_full_train), start=1):
    df_tr, df_va = df_full_train.iloc[tr_idx], df_full_train.iloc[va_idx]
    dv, model = train(df_tr, df_tr.is_canceled.values)
    auc = roc_auc_score(df_va.is_canceled.values, predict(df_va, dv, model))
    scores.append(auc)
    print(f"fold {i}/5: AUC = {auc:.4f}")
print(f"CV AUC: {np.mean(scores):.4f} +- {np.std(scores):.4f}")

# Model final
dv, model = train(df_full_train, df_full_train.is_canceled.values)
auc_test = roc_auc_score(df_test.is_canceled.values, predict(df_test, dv, model))
print(f"Test AUC: {auc_test:.4f}")

with open(OUTPUT_FILE, "wb") as f_out:
    pickle.dump((dv, model), f_out)
print(f"Model saved to {OUTPUT_FILE}")
