# 🎬 MCU Box Office Prediction

A machine learning project that predicts **worldwide box office revenue** for Marvel Cinematic Universe (MCU) movies using a Flask REST API backend and a Streamlit interactive frontend.

---

## 📁 Project Structure

```
MCU project 2/
│
├── data/
│   └── mcu.csv                  # Raw dataset (38 films + Disney+ shows)
│
├── models/
│   ├── best_model.pkl           # Trained ML model (auto-generated)
│   ├── label_encoder.pkl        # Phase label encoder (auto-generated)
│   └── model_meta.json          # Model metrics & metadata (auto-generated)
│
├── outputs/
│   ├── actual_vs_predicted.png  # Scatter plot (auto-generated)
│   ├── feature_importance.png   # Feature importance bar chart
│   ├── model_comparison.png     # R² comparison chart
│   ├── box_office_distribution.png
│   └── correlation_heatmap.png
│
├── train_model.py               # ML training script
├── app.py                       # Flask REST API backend
├── ui.py                        # Streamlit frontend
├── requirements.txt             # Python dependencies
└── README.md
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python train_model.py
```

This will:
- Load and clean `data/mcu.csv`
- Engineer features (phase encoding, release year, etc.)
- Train & compare LinearRegression, GradientBoosting, and RandomForest
- Save the best model to `models/best_model.pkl`
- Generate diagnostic plots in `outputs/`

### 3. Start the Flask Backend

```bash
python app.py
```

API will be available at **http://localhost:5000**

### 4. Launch the Streamlit Frontend

In a **new terminal**:

```bash
streamlit run ui.py
```

UI will open at **http://localhost:8501**

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check & model status |
| GET | `/api/model-info` | Model metadata & all metrics |
| GET | `/api/dataset` | Full MCU dataset as JSON |
| POST | `/api/predict` | Predict worldwide box office |
| GET | `/api/predictions/all` | Predictions for all movies |
| GET | `/api/feature-importance` | Feature importance scores |

### POST `/api/predict` – Request Body

```json
{
  "mcu_phase":          "Phase 4",
  "release_year":       2024,
  "tomato_meter":       85,
  "audience_score":     90,
  "movie_duration":     130,
  "production_budget":  200000000,
  "opening_weekend":    100000000,
  "domestic_box_office": 300000000
}
```

### Response

```json
{
  "predicted_worldwide_box_office": 850000000.0,
  "predicted_worldwide_box_office_millions": 850.0
}
```

---

## 📊 Dataset

- **Source:** [MCU Box Office Dataset on Kaggle](https://www.kaggle.com/datasets/parthfr/marvel-cinematic-universe-boxoffice-csv)
- **Total entries:** 63 (38 theatrical + 25 Disney+ shows)
- **Theatrical releases with full data:** 38 films
- **Target variable:** `worldwide_box_office`

### Columns

| Column | Description |
|--------|-------------|
| `movie_title` | Film title |
| `mcu_phase` | MCU narrative phase (Phase 1–6) |
| `release_date` | Theatrical release date |
| `tomato_meter` | Rotten Tomatoes critic score (%) |
| `audience_score` | Rotten Tomatoes audience score (%) |
| `movie_duration` | Runtime in minutes |
| `production_budget` | Production budget in USD |
| `opening_weekend` | Opening weekend domestic revenue in USD |
| `domestic_box_office` | Total US/Canada revenue in USD |
| `worldwide_box_office` | Total global revenue in USD (**target**) |

---

## 🤖 Machine Learning

### Features Used

1. `phase_encoded` – MCU phase as integer
2. `release_year` – Year of release
3. `tomato_meter` – Critic score
4. `audience_score` – Audience score
5. `movie_duration` – Runtime
6. `production_budget` – Budget
7. `opening_weekend` – Opening weekend revenue
8. `domestic_box_office` – Domestic total

### Models Evaluated

| Model | Description |
|-------|-------------|
| **LinearRegression** | Baseline linear model |
| **GradientBoostingRegressor** | Ensemble boosting (200 estimators) |
| **RandomForestRegressor** | Ensemble bagging (300 estimators) |

Best model is selected by **5-fold cross-validated R² score** and saved automatically.

---

## 🖥️ Frontend Pages

| Page | Description |
|------|-------------|
| 🏠 **Home** | Project overview, key stats, quick prediction |
| 📊 **Dataset** | Interactive dataset explorer with phase filter |
| 🔮 **Predict** | Full prediction form with ROI & comparison chart |
| 📈 **Analytics** | Actual vs Predicted scatter, feature importance, residuals |
| ℹ️ **Model Info** | Full model metrics, comparison table, raw JSON metadata |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Data Processing | pandas, numpy |
| Machine Learning | scikit-learn |
| Backend API | Flask 3.0, Flask-CORS |
| Frontend UI | Streamlit 1.35 |
| Visualisation | matplotlib, seaborn |
| Model Persistence | joblib |
| Report Generation | python-docx, Pillow |

---

## 📈 Sample Results

After training on all 38 theatrical releases:

- **Best Model:** RandomForestRegressor
- **R² Score:** ~0.99 (train set) / ~0.98 (test set)
- **Cross-Validated R²:** ~0.95
- **MAE:** ~$50–100 million

*Results may vary slightly depending on train/test split.*

---

## 📝 License

MIT License — free for educational and research use.
