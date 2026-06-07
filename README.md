# concrete_slump_test_using_SVR

A machine learning regression project that predicts the **compressive strength of concrete**
based on its mix ingredients using Support Vector Regression (SVR).

---

## Dataset

- **File:** `concrete_slump_test.csv`
- **Size:** 103 samples
- **Features:**

| Column | Description |
|--------|-------------|
| `Cement` | Cement content (kg/m³) |
| `Slag` | Slag content (kg/m³) |
| `Fly ash` | Fly ash content (kg/m³) |
| `Water` | Water content (kg/m³) |
| `SP` | Superplasticizer (kg/m³) |
| `Coarse Aggr.` | Coarse aggregate (kg/m³) |
| `Fine Aggr.` | Fine aggregate (kg/m³) |
| `SLUMP(cm)` | Slump test result |
| `FLOW(cm)` | Flow test result |

- **Target:** `Compressive Strength (Mpa)`

---

## 📊 Exploratory Data Analysis

- Annotated **correlation heatmap** (8×8) across all features and target
- Helps identify which ingredients most influence compressive strength

---

## Preprocessing

- Split: **70% training / 30% test** (`test_size=0.3, random_state=101`)
- Feature scaling with **StandardScaler** (fit on train, transform both)

---

## Model 1 — Base SVR (Default Parameters)

```python
SVR()  # default: kernel='rbf', C=1, epsilon=0.1
```

### Results

| Metric | Value |
|--------|-------|
| MAE | 3.56 MPa |
| RMSE | 5.12 MPa |
| Mean Target Value | 30.86 MPa |

> Error is ~11.6% of the mean — reasonable but improvable.

---

## Model 2 — GridSearchCV Tuned SVR

```python
param_grid = {
    'C':       [0.001, 0.01, 0.1, 0.5, 1],
    'kernel':  ['linear', 'poly', 'rbf'],
    'gamma':   ['scale', 'auto'],
    'degree':  [2, 3, 4],
    'epsilon': [0, 0.01, 0.1, 0.5, 1, 2]
}
```

- **5-fold cross-validation** (default cv=5)
- Searched across **540 parameter combinations**

### Best Parameters Found
### Results

| Metric | Base SVR | Tuned SVR | Improvement |
|--------|----------|-----------|-------------|
| MAE | 3.56 MPa | **1.85 MPa** | ↓ 48% |
| RMSE | 5.12 MPa | **2.32 MPa** | ↓ 55% |

> Tuning **cut the error in half** — a significant improvement.

---

## Libraries Used

- `numpy`, `pandas` — data handling
- `matplotlib`, `seaborn` — visualization
- `scikit-learn`:
  - `SVR`, `LinearSVR`
  - `StandardScaler`
  - `GridSearchCV`
  - `train_test_split`
  - `mean_absolute_error`, `mean_squared_error`, `r2_score`

---

## How to Run

1. Upload `concrete_slump_test.csv` to your Google Colab environment
2. Open the notebook in **Google Colab** or **Jupyter Notebook**
3. Run all cells in order

---

## Key Takeaways

| Step | Detail |
|------|--------|
| Best kernel | Linear |
| Best C | 1 |
| Best epsilon | 1 |
| Final MAE | **1.85 MPa** |
| Final RMSE | **2.32 MPa** |

> **Lesson:** Default SVR parameters are rarely optimal for regression tasks.
> Always tune `C`, `epsilon`, and `kernel` together —
> a linear kernel often outperforms RBF on structured tabular data.
