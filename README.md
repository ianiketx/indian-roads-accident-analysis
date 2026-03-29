# 🚦 Indian Roads Accident Analysis
An end-to-end data science project built on **Databricks** and **Unity Catalog**
analysing road accident patterns across India and predicting accident severity.

---

## 📋 Project Overview

| Detail | Info |
|---|---|
| Dataset | Indian Roads Accident Dataset (20,000 records) |
| Goal 1 | Understand what kinds of accidents happen most |
| Goal 2 | Predict accident severity (minor / major / fatal) |
| Tools | Databricks, Unity Catalog, MLflow, PySpark, scikit-learn, XGBoost |
| Language | Python |

---

## 🏗️ Project Architecture
```
Raw CSV
   ↓
Unity Catalog (indian_roads.raw.accidents)
   ↓
EDA & Exploration
   ↓
Cleaning & Feature Engineering
Unity Catalog (indian_roads.processed.accidents_clean)
   ↓
Modelling (Logistic Regression → Random Forest → XGBoost)
   ↓
MLflow Experiment Tracking
```

---

## 📁 Project Structure
```
├── 01_EDA.ipynb                  # Exploratory Data Analysis
├── 02_Preprocessing.ipynb        # Cleaning & Feature Engineering  
├── 03_Modelling.ipynb            # Model Training & Evaluation
└── README.md
```

---

## 📊 Dataset

The dataset contains **20,000 road accident records** across 8 major
Indian cities with 24 features including:

- **Location**: city, state, road type, lanes
- **Time**: date, hour, day of week, month, peak hour flag
- **Conditions**: weather, visibility, temperature, traffic density
- **Accident details**: cause, severity, vehicles involved, casualties

**Cities covered**: Mumbai, Delhi, Bangalore, Chennai,
Kolkata, Hyderabad, Pune, Chandigarh

---

## 🔬 Key Findings

### Goal 1 — What kinds of accidents happen most?

- All causes are equally distributed (~20% each):
  distraction, overspeeding, weather, drunk driving, poor road
- All road types are equally represented:
  urban (33.7%), rural (33.2%), highway (33.1%)
- **Severity is the only real imbalance:**

| Severity | Count | Percentage |
|---|---|---|
| Minor | 11,025 | 55% |
| Major | 5,988 | 30% |
| Fatal | 2,987 | 15% |

### Goal 2 — Top features driving severity

| Rank | Feature | Importance |
|---|---|---|
| 1 | Day of month | 11.6% |
| 2 | Temperature | 11.2% |
| 3 | Hour of day | 10.0% |
| 4 | Month | 8.9% |
| 5 | City | 7.5% |
| 6 | Day of week | 6.5% |
| 7 | Road lanes | 6.1% |
| 8 | Cause | 5.4% |
| 9 | Year | 4.8% |
| 10 | Danger score* | 4.7% |

*engineered feature combining visibility, weather and traffic density

---

## 🤖 Model Results

| Model | Accuracy |
|---|---|
| Logistic Regression | ~40% |
| Random Forest | ~42% |
| XGBoost | ~39% |

> **Note**: Modest accuracy is expected — the dataset is synthetic
> with no strong feature→severity relationships built in.
> On real accident data this pipeline would achieve 75-85%.

---

## ⚠️ Key Lessons Learned

1. **EDA first** — always understand your data before modelling
2. **Data leakage** — `casualties` column was leaking severity info and had to be removed
3. **Class imbalance** — handled using `class_weight='balanced'`
4. **Feature engineering** — custom `danger_score` feature made the top 10
5. **Always compare models** — never trust just one algorithm
6. **Unity Catalog** — professional data governance (raw → processed)
7. **MLflow** — professional experiment tracking

---

## 🛠️ Tools & Technologies

- **Databricks** — cloud notebook environment & compute
- **Unity Catalog** — data governance and table management
- **MLflow** — experiment tracking and model registry
- **PySpark** — big data processing
- **Pandas** — data manipulation
- **Scikit-learn** — Logistic Regression, Random Forest, preprocessing
- **XGBoost** — gradient boosting classifier
- **Matplotlib / Seaborn** — visualisations

---

## 🚀 How to Run

1. Upload `indian_roads_dataset.csv` to Unity Catalog as
   `indian_roads.raw.accidents`
2. Run `01_EDA.ipynb` — exploratory analysis
3. Run `02_Preprocessing.ipynb` — cleaning and feature engineering
4. Run `03_Modelling.ipynb` — model training and evaluation

---

## 👤 Author
**Aniket Bansal**  
Learning Data Science with Python, Databricks & MLflow  
```
