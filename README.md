# Housing Price ML — Airbnb Madrid Price Predictor

Machine learning model that predicts the nightly price of an Airbnb listing in Madrid based on property features.

---

## Goal

Train and compare four ML models to predict listing prices and identify the key drivers of price in the Madrid short-term rental market.

## Project structure

```
housing-price-ml/
├── data/
│   └── raw/                    ← listings.csv.gz (Inside Airbnb)
├── notebooks/
│   └── 02_housing_price_ml.ipynb   ← main notebook
├── src/
│   ├── features.py             ← feature engineering pipeline
│   └── evaluate.py             ← metrics and plotting utilities
├── models/
│   └── best_price_model.joblib ← saved best model
├── outputs/
│   └── figures/                ← auto-generated charts
├── requirements.txt
└── README.md
```

---

## Models compared

| Model | Description |
|-------|-------------|
| Ridge Regression | Linear baseline with L2 regularisation |
| Random Forest | Ensemble of 200 decision trees |
| Gradient Boosting | Sequential boosting (sklearn) |
| **XGBoost** | Optimised gradient boosting — best performer |

---

## Features used

- Capacity (`accommodates`, `bedrooms`, `beds`, `bathrooms`)
- Location (`neighbourhood_cleansed`, `latitude`, `longitude`, neighbourhood median price)
- Listing type (`room_type`)
- Host info (`host_is_superhost`, `host_listings_count`)
- Availability & reviews

---

## Installation & usage

```bash
git clone https://github.com/Abadalina/housing-price-ml.git
cd housing-price-ml
pip install -r requirements.txt
jupyter notebook notebooks/02_housing_price_ml.ipynb
```

---

## Results

| Model | MAE | RMSE | R2 |
|-------|-----|------|----|
| Ridge Regression | — | — | — |
| Random Forest | — | — | — |
| Gradient Boosting | — | — | — |
| XGBoost | — | — | **—** |

> Run the notebook to populate this table with real values.

---

## Tech stack

![Python](https://img.shields.io/badge/Python-3.11-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-F7931E)
![XGBoost](https://img.shields.io/badge/XGBoost-2.1-189B3A)
![Pandas](https://img.shields.io/badge/Pandas-2.2-150458)

---

## Author

**Alejandro Abadal** — Data Science Student, UOC
[LinkedIn](#) · [GitHub](#)

---

*Data for educational purposes. Source: Inside Airbnb.*
