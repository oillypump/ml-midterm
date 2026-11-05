# Hotel Booking Cancellation Prediction

Midterm project for [ML Zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp).

## Problem description

Booking cancellations are a major source of lost revenue for hotels. A room that was
expected to be occupied ends up empty, often too late to be resold. If a hotel knows
**which bookings are likely to be cancelled**, it can act early, for example:

- send a reminder or ask for a deposit on high-risk bookings,
- offer the room to a waiting list,
- plan overbooking more carefully,
- forecast occupancy and revenue more realistically.

This project builds a **binary classifier** that predicts whether a booking will be
cancelled (`is_canceled = 1`) from information known at booking time (lead time, market
segment, price, number of guests, special requests, guest history, etc.).

The model is exposed as a web service. A hotel system sends the booking details as JSON
and receives the **cancellation probability** and a decision (threshold 0.5, adjustable
by the hotel depending on the cost of false positives vs. false negatives).

## Dataset

[Hotel Booking Cancellation Prediction (Kaggle)](https://www.kaggle.com/datasets/youssefaboelwafa/hotel-booking-cancellation-prediction)

- 36,285 bookings, 17 columns, no missing values
- Target: `booking status` (`Canceled` / `Not_Canceled`), about 32.8% cancelled
- Features: number of adults/children, weekend/week nights, meal plan, parking, room type,
  lead time, market segment, repeated guest, previous cancellations (`P-C`),
  previous non-cancelled bookings (`P-not-C`), average price, special requests,
  date of reservation

### Getting the data

```bash
mkdir -p data
curl -L -o data/hotel-booking-cancellation-prediction.zip \
  https://www.kaggle.com/api/v1/datasets/download/youssefaboelwafa/hotel-booking-cancellation-prediction
unzip -o data/hotel-booking-cancellation-prediction.zip -d data/
```

The file used by the notebook and `train.py` is `data/booking.csv`.

<!-- TODO: if the dataset is committed to the repo, replace this section accordingly -->

## Exploratory data analysis (summary)

All target-related analysis was done on the training portion only (80%); the test set
was kept untouched until the final evaluation.

- Global cancellation rate: 32.8%.
- `lead_time` is the strongest predictor and its relationship is non-linear:
  8.7% cancellation at ≤7 days up to 72.6% at 181–365 days (correlation with target 0.44).
- `special_requests` is negatively correlated with cancellation (-0.26).
- Guests with booking history (`p_c` or `p_not_c` > 0) almost never cancel (<5%).
- Market segment matters: online 36%, offline 30%, corporate 11%, complementary 0%.
- Month of reservation shows large variation, but the data covers roughly Oct 2017 to
  Sep 2018 and the monthly pattern moves together with average lead time, so seasonality
  cannot be confirmed.
- Two rows dated 2015/2016 were treated as anomalies; the invalid date `2018-2-29` was
  corrected to `2018-02-28`.

## Modeling

Features: `type_of_meal`, `room_type`, `market_segment_type`, `res_month` and 11 numeric
columns. `res_dow` was dropped (no measurable gain), and `res_month` was kept because
removing it lowered validation AUC (0.950 → 0.940).

Three models were tuned with 5-fold cross-validation on the training data
(`df_full_train`); parameters were first explored on a single validation split and then
confirmed with K-Fold.

| Model               | Tuned parameters                                 | Selected                        | CV AUC (mean ± std) |
| ------------------- | ------------------------------------------------ | ------------------------------- | ------------------- |
| Logistic Regression | `C`                                              | `C=1`                           | 0.8706 ± 0.0035     |
| Random Forest       | `max_depth`, `min_samples_leaf`, `n_estimators`  | 20, 1, 100                      | 0.9505 ± 0.0015     |
| **XGBoost**         | `max_depth`, `learning_rate`, `min_child_weight` | 10, 0.2, 1 (`n_estimators=100`) | **0.9526 ± 0.0024** |

XGBoost and Random Forest are statistically close (difference smaller than the standard
deviation); XGBoost was chosen for the highest mean AUC and consistent results across
both evaluation methods. Logistic Regression is clearly worse, mainly because it cannot
capture the non-linear effect of `lead_time`.

### Final evaluation (test set, evaluated once)

| Metric    | Value (threshold 0.5) |
| --------- | --------------------- |
| ROC AUC   | 0.9548                |
| Precision | 0.866                 |
| Recall    | 0.802                 |
| F1        | 0.833                 |

Lowering the threshold increases recall (0.875 at 0.3) at the cost of precision (0.786).
The right threshold depends on the hotel's cost of missed cancellations vs. false alarms.

### Limitations

- Random train/validation/test split; performance on a future period may be lower.
- Only about 15 months of data, so seasonal effects are not reliable.
- Tree models can output very extreme probabilities; the ranking of risk is more
  trustworthy than the exact probability value.

## Project structure

```
.
├── README.md
├── notebook.ipynb      # data preparation, EDA, model selection and tuning
├── train.py            # trains the final model and saves model.bin
├── predict.py          # web service (TODO: Flask/FastAPI)
├── predict-test.py     # example request to the service (TODO)
├── requirements.txt    # dependencies (TODO)
├── Dockerfile          # (TODO)
├── model.bin           # saved DictVectorizer + XGBoost model
└── data/
    └── booking.csv
```

## How to run

### 1. Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

<!-- TODO: adjust if you use pipenv / uv -->

### 2. Train the model

```bash
python train.py
```

This runs 5-fold CV, trains the final model on the full training split, prints the test
AUC and saves `model.bin`. Expected: CV AUC ≈ 0.9526 and test AUC ≈ 0.9548.

### 3. Run the web service locally

```bash
# TODO: replace with your actual command, e.g.
# python predict.py
# or: uvicorn predict:app --host 0.0.0.0 --port 9696
```

### 4. Test the service

```bash
python predict-test.py
```

Example request body:

```json
{
  "type_of_meal": "meal_plan_1",
  "room_type": "room_type_1",
  "market_segment_type": "online",
  "res_month": "7",
  "number_of_adults": 2,
  "number_of_children": 0,
  "number_of_weekend_nights": 1,
  "number_of_week_nights": 2,
  "car_parking_space": 0,
  "lead_time": 200,
  "repeated": 0,
  "p_c": 0,
  "p_not_c": 0,
  "average_price": 110.0,
  "special_requests": 0
}
```

Categorical values must match the training data exactly (lowercase, e.g.
`meal_plan_1`, `room_type_1`, `online`), and `res_month` is a string (`"1"`–`"12"`).

<!-- TODO: add example response format once predict.py is done -->

### 5. Docker

```bash
# TODO: confirm image name and port
docker build -t hotel-cancellation .
docker run -it --rm -p 9696:9696 hotel-cancellation
```

### 6. Cloud deployment

<!-- TODO: platform, deployment steps, public URL or screenshot/video -->

## Reproducibility

- Dataset download instructions are above (or the dataset is committed in `data/`).
- The notebook runs top to bottom (Restart & Run All) without errors.
- `train.py` runs end to end from the raw CSV.
- Fixed random seeds (`random_state=1`) are used for splits, K-Fold and models.
