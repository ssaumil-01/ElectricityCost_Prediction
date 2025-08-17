# Electricity Cost Prediction (ML + Flask)

Predict electricity cost for urban structures using a clean, end‑to‑end machine‑learning pipeline wrapped in a lightweight Flask web app.

---

## ✨ Highlights

- **End‑to‑end pipeline**: Robust preprocessing (imputation, scaling, one‑hot) + Random Forest regressor, tuned via cross‑validation.
- **Simple UI & API**: Form-driven web interface with a `POST /predict` endpoint.
- **Strong results**: High explanatory power and low error on held‑out data.
- **Reproducible**: Encapsulated `Pipeline` ensures train/infer parity.

---

## 🧠 Problem

Estimate the **monthly electricity cost** of a building/structure from site, usage, and environmental attributes. Typical use cases include facility planning, bill forecasting, and what‑if analysis for building managers.

---

## 🗂️ Data (at a glance)

- **Rows / Columns**: ~10k / 9  
- **Features**: Mixed numeric (e.g., *site_area*, *water_consumption*, *recycling_rate*, *utilisation_rate*, *air_quality_index*, *issue_resolution_time*, *resident_count*) + one categorical (*structure_type*: Residential | Mixed‑use | Commercial).  
- **Target**: Monthly electricity cost (₹).

**EDA nuggets** (examples):
- Cost spans roughly **₹500–₹6,446** with an average near **₹2,800**.
- Positive correlation with **site area**, **resident count**, and **water consumption**.
- **Mixed‑use** structures tend to incur higher costs than purely residential/commercial.

> Tip: Keep input ranges realistic; the model was trained on values in the domains above.

---

## 🏗️ Approach

### Preprocessing
- **Numerical**: median imputation → `StandardScaler`
- **Categorical**: most‑frequent imputation → `OneHotEncoder(handle_unknown="ignore")`
- All steps are composed in a `ColumnTransformer` inside a single `Pipeline` to keep transformations identical during training and inference.

### Model
- **Estimator**: `RandomForestRegressor`
- **Tuning**: `GridSearchCV` (3‑fold) over:
  - `n_estimators`: 100, 200
  - `max_depth`: None, 10, 20
  - `min_samples_split`: 2, 5
  - `min_samples_leaf`: 1, 2

### Performance (test set)
| Metric | Score |
|---|---|
| R² | **0.96** |
| RMSE | **≈ ₹221** |
| MSE | **48,800.04** |

> Interpretation: The model explains ~96% of target variance and yields an average absolute error on the order of a few hundred rupees.

---

## 🧪 Minimal Repro Code (illustrative)

```python
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split

# Example columns — adjust to your dataset
num_cols = [
    "site_area", "water_consumption", "recycling_rate",
    "utilisation_rate", "air_quality_index", "issue_resolution_time",
    "resident_count"
]
cat_cols = ["structure_type"]

preprocess = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), num_cols),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore"))
    ]), cat_cols),
])

model = RandomForestRegressor(random_state=42)

pipe = Pipeline([("prep", preprocess), ("rf", model)])

param_grid = {
    "rf__n_estimators": [100, 200],
    "rf__max_depth": [None, 10, 20],
    "rf__min_samples_split": [2, 5],
    "rf__min_samples_leaf": [1, 2],
}

gcv = GridSearchCV(pipe, param_grid, cv=3, scoring="neg_mean_squared_error", n_jobs=-1)

# df = pd.read_csv("data.csv")  # Load your data
# X = df[num_cols + cat_cols]
# y = df["electricity_cost"]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# gcv.fit(X_train, y_train)
# print(gcv.best_params_, gcv.best_score_)
```

---

## 🖥️ App

### Stack
- **Backend**: Flask
- **ML**: scikit‑learn (pipeline + inference)
- **Data/Utils**: pandas, numpy

### Endpoints
- `GET /` → Renders the input form (`templates/index.html`).
- `POST /predict` → Accepts form fields and returns a formatted prediction.

#### Expected form fields
```
site_area (int)
water_consumption (float)
recycling_rate (int)
utilisation_rate (int)
air_quality_index (int)
issue_resolution_time (int)
resident_count (int)
structure_type (str: Residential | Mixed-use | Commercial)
```

#### Example (cURL form POST)
```bash
curl -X POST http://127.0.0.1:5000/predict   -H "Content-Type: application/x-www-form-urlencoded"   -d "site_area=1500"   -d "water_consumption=320.5"   -d "recycling_rate=40"   -d "utilisation_rate=76"   -d "air_quality_index=92"   -d "issue_resolution_time=3"   -d "resident_count=24"   -d "structure_type=Mixed-use"
```

---

## 🏁 Getting Started

### 1) Environment
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
# If no requirements.txt yet, minimally:
pip install flask scikit-learn pandas numpy
```

### 3) Run the app
```bash
# Option A
python app.py

# Option B
export FLASK_APP=app.py && flask run     # macOS/Linux
set FLASK_APP=app.py && flask run        # Windows (cmd)
$env:FLASK_APP="app.py"; flask run       # Windows (PowerShell)
```

Then visit `http://127.0.0.1:5000/`.

---

## 📦 Suggested Repo Layout

```
.
├─ app.py
├─ requirements.txt
├─ model/                # trained model artifact(s)
├─ notebooks/            # EDA & training
├─ data/                 # raw / processed data (or add a data link)
├─ templates/
│  └─ index.html         # form UI
├─ static/               # (optional) styles/scripts/assets
└─ README.md
```

> Adjust names/paths as needed to match your codebase.

---

## 📊 Understanding & Responsible Use

- **Intended use**: Cost forecasting and comparative analysis within the same data distribution.
- **Limitations**: Out‑of‑distribution inputs (e.g., unseen structure types, extreme magnitudes) may degrade accuracy despite `handle_unknown="ignore"` in the encoder.
- **Data quality**: Predictions depend on valid ranges and consistent units for numeric fields.
- **Future work**: Add uncertainty bands, time‑aware features, SHAP explanations, automated retraining, and containerization.

---

## 🤝 Contributing

Issues and PRs are welcome! Please open an issue to discuss major changes.

---

## 📄 License

Add your chosen license (e.g., MIT) here.
