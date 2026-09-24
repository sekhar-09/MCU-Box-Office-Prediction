"""
main.py  -  MCU Box Office Prediction  (all-in-one)
============================================================
Usage:
  python main.py train    ->  train the ML model & save artefacts
  python main.py api      ->  start the Flask REST API (port 5000)
  python main.py ui       ->  launch the Streamlit frontend (port 8501)
  python main.py all      ->  train first, then start the Flask API

Streamlit note:
  Streamlit must be launched via its own CLI, so `python main.py ui`
  re-invokes the script through `streamlit run main.py --ui-mode` automatically.
============================================================
"""

import os
import sys

# ── Shared paths (used by all three modules) ───────────────────────────────────
DATA_PATH  = os.path.join("data", "mcu.csv")
MODEL_DIR  = "models"
OUTPUT_DIR = "outputs"
os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
LE_PATH    = os.path.join(MODEL_DIR, "label_encoder.pkl")
META_PATH  = os.path.join(MODEL_DIR, "model_meta.json")

API_BASE   = "http://localhost:5000/api"


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 – TRAINING  (python main.py train)
# ══════════════════════════════════════════════════════════════════════════════
def run_training():
    import json
    import joblib
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.preprocessing import LabelEncoder

    # ── Load & clean ───────────────────────────────────────────────────────────
    def load_data(path):
        df = pd.read_csv(path)
        df = df[df["worldwide_box_office"].notna() & (df["worldwide_box_office"] != "NA")]
        for col in ["tomato_meter", "audience_score", "movie_duration",
                    "production_budget", "opening_weekend",
                    "domestic_box_office", "worldwide_box_office"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df.dropna(subset=["worldwide_box_office"], inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    # ── Feature engineering ────────────────────────────────────────────────────
    def engineer_features(df):
        le = LabelEncoder()
        df = df.copy()
        df["phase_encoded"] = le.fit_transform(df["mcu_phase"])
        df["release_year"]  = pd.to_datetime(df["release_date"], errors="coerce").dt.year
        FEATURES = [
            "phase_encoded", "release_year", "tomato_meter", "audience_score",
            "movie_duration", "production_budget", "opening_weekend", "domestic_box_office",
        ]
        model_df = df[FEATURES + ["worldwide_box_office"]].dropna()
        return model_df[FEATURES], model_df["worldwide_box_office"], FEATURES, le

    # ── Train & evaluate ───────────────────────────────────────────────────────
    def train_and_evaluate(X, y, features):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        candidates = {
            "LinearRegression": LinearRegression(),
            "GradientBoosting": GradientBoostingRegressor(n_estimators=200, random_state=42),
            "RandomForest":     RandomForestRegressor(n_estimators=300, random_state=42),
        }
        results = {}
        for name, model in candidates.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            mae   = mean_absolute_error(y_test, preds)
            rmse  = np.sqrt(mean_squared_error(y_test, preds))
            r2    = r2_score(y_test, preds)
            cv    = cross_val_score(model, X, y, cv=5, scoring="r2")
            results[name] = {
                "model": model, "MAE": mae, "RMSE": rmse, "R2": r2,
                "CV_R2_mean": cv.mean(), "CV_R2_std": cv.std(),
                "y_test": y_test, "preds": preds,
            }
            print(f"  [{name}]  MAE={mae/1e6:.1f}M  RMSE={rmse/1e6:.1f}M  "
                  f"R2={r2:.3f}  CV-R2={cv.mean():.3f}+-{cv.std():.3f}")
        best_name = max(results, key=lambda k: results[k]["CV_R2_mean"])
        print(f"\n  [OK] Best model: {best_name}")
        return results, best_name

    # ── Save artefacts ─────────────────────────────────────────────────────────
    def save_artefacts(results, best_name, features, le):
        best = results[best_name]
        joblib.dump(best["model"], MODEL_PATH)
        joblib.dump(le,            LE_PATH)
        meta = {
            "best_model": best_name,
            "features":   features,
            "metrics": {
                "MAE": best["MAE"], "RMSE": best["RMSE"], "R2": best["R2"],
                "CV_R2_mean": best["CV_R2_mean"], "CV_R2_std": best["CV_R2_std"],
            },
            "all_models": {
                k: {"MAE": v["MAE"], "RMSE": v["RMSE"], "R2": v["R2"],
                    "CV_R2_mean": v["CV_R2_mean"], "CV_R2_std": v["CV_R2_std"]}
                for k, v in results.items()
            },
        }
        with open(META_PATH, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"  Model saved -> {MODEL_PATH}")
        return meta

    # ── Diagnostic plots ───────────────────────────────────────────────────────
    def make_plots(results, best_name, X, y, features):
        best = results[best_name]

        # Actual vs Predicted
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(best["y_test"] / 1e6, best["preds"] / 1e6,
                   color="#3b82d4", edgecolors="white", s=80, alpha=0.85)
        mn = min(best["y_test"].min(), best["preds"].min()) / 1e6
        mx = max(best["y_test"].max(), best["preds"].max()) / 1e6
        ax.plot([mn, mx], [mn, mx], "r--", lw=1.5, label="Perfect fit")
        ax.set_xlabel("Actual ($ millions)"); ax.set_ylabel("Predicted ($ millions)")
        ax.set_title(f"Actual vs Predicted - {best_name}"); ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, "actual_vs_predicted.png"), dpi=120)
        plt.close(fig)

        # Feature importance
        model = best["model"]
        if hasattr(model, "feature_importances_"):
            imp = pd.Series(model.feature_importances_, index=features).sort_values()
            fig, ax = plt.subplots(figsize=(7, 4))
            imp.plot(kind="barh", color="#3b82d4", ax=ax)
            ax.set_title("Feature Importances"); ax.set_xlabel("Importance")
            fig.tight_layout()
            fig.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"), dpi=120)
            plt.close(fig)

        # Model comparison
        names  = list(results.keys())
        r2vals = [results[n]["R2"] for n in names]
        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(names, r2vals, color=["#3b82d4", "#7c5cd8", "#10b981"])
        ax.set_ylim(0, 1.05); ax.set_ylabel("R2 Score")
        ax.set_title("Model Comparison - R2 on Test Set")
        for bar, val in zip(bars, r2vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01, f"{val:.3f}", ha="center", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, "model_comparison.png"), dpi=120)
        plt.close(fig)

        # Distribution
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(y / 1e6, bins=15, color="#3b82d4", edgecolor="white")
        ax.set_xlabel("Worldwide Box Office ($ millions)"); ax.set_ylabel("Count")
        ax.set_title("Distribution of Worldwide Box Office")
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, "box_office_distribution.png"), dpi=120)
        plt.close(fig)

        # Correlation heatmap
        corr_df = X.copy(); corr_df["worldwide_box_office"] = y.values
        corr_matrix = corr_df.corr().values
        labels = list(corr_df.columns); n = len(labels)
        fig, ax = plt.subplots(figsize=(9, 7))
        im = ax.imshow(corr_matrix, cmap="Blues", vmin=-1, vmax=1)
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{corr_matrix[i, j]:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if abs(corr_matrix[i, j]) > 0.6 else "black")
        ax.set_title("Feature Correlation Heatmap")
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"), dpi=120)
        plt.close(fig)

        print(f"  Plots saved -> {OUTPUT_DIR}/")

    # ── Run ────────────────────────────────────────────────────────────────────
    print("=" * 50)
    print("  MCU Box Office Prediction - Training")
    print("=" * 50)
    df = load_data(DATA_PATH)
    print(f"  Loaded {len(df)} theatrical releases.\n")
    X, y, features, le = engineer_features(df)
    print(f"  Features : {features}")
    print(f"  Samples  : {len(X)}\n")
    results, best_name = train_and_evaluate(X, y, features)
    save_artefacts(results, best_name, features, le)
    make_plots(results, best_name, X, y, features)
    print("\n  Training complete.\n")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 – FLASK API  (python main.py api)
# ══════════════════════════════════════════════════════════════════════════════
def run_api():
    import json
    import joblib
    import numpy as np
    import pandas as pd
    from flask import Flask, request, jsonify
    from flask_cors import CORS

    app = Flask(__name__)
    CORS(app)

    # Load saved artefacts
    def _load():
        model = joblib.load(MODEL_PATH)
        le    = joblib.load(LE_PATH)
        with open(META_PATH) as f:
            meta = json.load(f)
        return model, le, meta

    try:
        MODEL, LE, META = _load()
        FEATURES = META["features"]
        print("[OK] Model loaded successfully.")
    except Exception as exc:
        MODEL = LE = META = FEATURES = None
        print(f"[WARN] Model not loaded: {exc}. Run: python main.py train")

    def _get_df():
        df = pd.read_csv(DATA_PATH)
        df = df[df["worldwide_box_office"].notna() & (df["worldwide_box_office"] != "NA")]
        for col in ["tomato_meter", "audience_score", "movie_duration",
                    "production_budget", "opening_weekend",
                    "domestic_box_office", "worldwide_box_office"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["release_year"]  = pd.to_datetime(df["release_date"], errors="coerce").dt.year
        df["phase_encoded"] = LE.transform(df["mcu_phase"])
        return df

    @app.route("/api/health", methods=["GET"])
    def health():
        return jsonify({
            "status":       "ok",
            "model_loaded": MODEL is not None,
            "model_name":   META["best_model"] if META else None,
        })

    @app.route("/api/model-info", methods=["GET"])
    def model_info():
        if not META:
            return jsonify({"error": "Model not trained yet"}), 503
        return jsonify({
            "best_model": META["best_model"],
            "features":   META["features"],
            "metrics":    {k: round(v, 4) for k, v in META["metrics"].items()},
            "all_models": {
                name: {k: round(v, 4) for k, v in vals.items()}
                for name, vals in META["all_models"].items()
            },
        })

    @app.route("/api/dataset", methods=["GET"])
    def dataset():
        df = pd.read_csv(DATA_PATH)
        df = df.where(pd.notnull(df), None)
        df["release_year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year
        return jsonify({
            "total_entries":    len(df),
            "theatrical_count": int(df["worldwide_box_office"].notna().sum()),
            "columns":          list(df.columns),
            "data":             df.to_dict(orient="records"),
        })

    @app.route("/api/predict", methods=["POST"])
    def predict():
        if not MODEL:
            return jsonify({"error": "Model not trained yet"}), 503
        body = request.get_json(force=True)
        required = ["mcu_phase", "release_year", "tomato_meter", "audience_score",
                    "movie_duration", "production_budget", "opening_weekend", "domestic_box_office"]
        missing = [k for k in required if k not in body]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400
        try:
            phase_enc = int(LE.transform([body["mcu_phase"]])[0])
        except Exception:
            return jsonify({"error": f"Unknown phase. Valid: {list(LE.classes_)}"}), 400
        row = np.array([[
            phase_enc,
            float(body["release_year"]),
            float(body["tomato_meter"]),
            float(body["audience_score"]),
            float(body["movie_duration"]),
            float(body["production_budget"]),
            float(body["opening_weekend"]),
            float(body["domestic_box_office"]),
        ]])
        prediction = float(MODEL.predict(row)[0])
        return jsonify({
            "input":                                   body,
            "predicted_worldwide_box_office":          round(prediction, 2),
            "predicted_worldwide_box_office_millions": round(prediction / 1e6, 2),
        })

    @app.route("/api/predictions/all", methods=["GET"])
    def predictions_all():
        if not MODEL:
            return jsonify({"error": "Model not trained yet"}), 503
        df = _get_df()
        feat_df = df[FEATURES].dropna()
        valid   = df.loc[feat_df.index].copy()
        preds   = MODEL.predict(feat_df)
        records = []
        for idx, row in valid.iterrows():
            pred = preds[list(valid.index).index(idx)]
            records.append({
                "movie_title":                    row["movie_title"],
                "mcu_phase":                      row["mcu_phase"],
                "release_year":                   int(row["release_year"]) if pd.notna(row["release_year"]) else None,
                "actual_worldwide_box_office":    row["worldwide_box_office"],
                "predicted_worldwide_box_office": round(float(pred), 2),
                "difference_millions":            round((float(pred) - float(row["worldwide_box_office"])) / 1e6, 2),
            })
        return jsonify({"count": len(records), "predictions": records})

    @app.route("/api/feature-importance", methods=["GET"])
    def feature_importance():
        if not MODEL:
            return jsonify({"error": "Model not trained yet"}), 503
        if hasattr(MODEL, "feature_importances_"):
            imp  = dict(zip(FEATURES, MODEL.feature_importances_.tolist()))
            kind = "importance"
        elif hasattr(MODEL, "coef_"):
            coefs = np.abs(MODEL.coef_)
            imp   = dict(zip(FEATURES, (coefs / (coefs.sum() or 1.0)).tolist()))
            kind  = "normalised_abs_coefficient"
        else:
            return jsonify({"error": "Model does not expose feature weights"}), 400
        sorted_imp = dict(sorted(imp.items(), key=lambda x: x[1], reverse=True))
        return jsonify({"kind": kind, "feature_importance": sorted_imp})

    print("=" * 50)
    print("  MCU Box Office Prediction - Flask API")
    print("  Running on http://localhost:5000")
    print("=" * 50)
    app.run(host="0.0.0.0", port=5000, debug=False)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 – STREAMLIT UI  (python main.py ui  →  streamlit run main.py)
# ══════════════════════════════════════════════════════════════════════════════
def run_ui():
    """Re-launch this file via `streamlit run`, then exit."""
    import subprocess
    subprocess.run([sys.executable, "-m", "streamlit", "run", __file__, "--", "--ui-mode"],
                   check=True)


def _streamlit_app():
    """Called when Streamlit imports this file directly."""
    import requests
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import streamlit as st

    st.set_page_config(
        page_title="MCU Box Office Predictor",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown("""
    <style>
        .main-header   { font-size:2.4rem; font-weight:700; color:#3b82d4; text-align:center; }
        .sub-header    { font-size:1.1rem; color:#57606a; text-align:center; margin-bottom:2rem; }
        .metric-card   { background:#f7f8fa; border:1px solid #e5e7eb; border-radius:10px;
                         padding:1.2rem; text-align:center; }
        .metric-value  { font-size:1.8rem; font-weight:700; color:#3b82d4; }
        .metric-label  { font-size:0.85rem; color:#57606a; }
        .pred-result   { background:linear-gradient(135deg,#3b82d4,#7c5cd8); color:white;
                         border-radius:12px; padding:2rem; text-align:center;
                         font-size:2rem; font-weight:700; }
    </style>
    """, unsafe_allow_html=True)

    # ── Cached API helpers ─────────────────────────────────────────────────────
    @st.cache_data(ttl=60)
    def get_health():
        try: return requests.get(f"{API_BASE}/health", timeout=5).json()
        except: return None

    @st.cache_data(ttl=120)
    def get_model_info():
        try: return requests.get(f"{API_BASE}/model-info", timeout=5).json()
        except: return None

    @st.cache_data(ttl=300)
    def get_dataset():
        try: return requests.get(f"{API_BASE}/dataset", timeout=5).json()
        except: return None

    @st.cache_data(ttl=300)
    def get_predictions_all():
        try: return requests.get(f"{API_BASE}/predictions/all", timeout=5).json()
        except: return None

    @st.cache_data(ttl=300)
    def get_feature_importance():
        try: return requests.get(f"{API_BASE}/feature-importance", timeout=5).json()
        except: return None

    def fmt_billions(val):
        return "N/A" if val is None else f"${val/1e9:.2f}B"

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🎬 MCU Predictor")
        st.markdown("---")
        page = st.radio("Navigation",
                        ["🏠 Home", "📊 Dataset", "🔮 Predict", "📈 Analytics", "ℹ️ Model Info"])
        st.markdown("---")
        health = get_health()
        if health and health.get("status") == "ok":
            st.success("API Online")
            st.caption(f"Model: **{health.get('model_name','N/A')}**")
        else:
            st.error("API Offline")
            st.caption("Run: python main.py api")
        st.markdown("---")
        st.caption("MCU Box Office Predictor\nFlask + Streamlit + scikit-learn")

    # ── HOME ───────────────────────────────────────────────────────────────────
    if page == "🏠 Home":
        st.markdown('<div class="main-header">🎬 MCU Box Office Predictor</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Machine Learning Model to Predict Marvel Movie Worldwide Revenue</div>', unsafe_allow_html=True)
        dataset = get_dataset(); model = get_model_info()
        col1, col2, col3, col4 = st.columns(4)
        if dataset:
            theatrical = [r for r in dataset["data"] if r.get("worldwide_box_office")]
            total_ww   = sum(r["worldwide_box_office"] for r in theatrical if r["worldwide_box_office"])
            col1.markdown(f'<div class="metric-card"><div class="metric-value">{len(theatrical)}</div><div class="metric-label">Theatrical Releases</div></div>', unsafe_allow_html=True)
            col2.markdown(f'<div class="metric-card"><div class="metric-value">{fmt_billions(total_ww)}</div><div class="metric-label">Total WW Box Office</div></div>', unsafe_allow_html=True)
        if model:
            col3.markdown(f'<div class="metric-card"><div class="metric-value">{model["metrics"]["R2"]:.3f}</div><div class="metric-label">Model R²</div></div>', unsafe_allow_html=True)
            col4.markdown(f'<div class="metric-card"><div class="metric-value">${model["metrics"]["MAE"]/1e6:.0f}M</div><div class="metric-label">Mean Abs Error</div></div>', unsafe_allow_html=True)
        st.markdown("---")
        col_l, col_r = st.columns([3, 2])
        with col_l:
            st.subheader("About This Project")
            st.markdown("""
Predicts **worldwide box office revenue** for MCU films using machine learning.

**Features:** MCU Phase · Release Year · RT Score · Audience Score ·
Duration · Budget · Opening Weekend · Domestic Box Office

**Models:** Linear Regression · Gradient Boosting · Random Forest

**Stack:** Python · scikit-learn · Flask · Streamlit
""")
        with col_r:
            st.subheader("Quick Prediction")
            if health and health.get("status") == "ok":
                budget  = st.number_input("Budget ($M)",          100, 500, 200, step=10)
                opening = st.number_input("Opening Weekend ($M)",  10,  400, 100, step=5)
                tomato  = st.slider("Rotten Tomatoes %", 0, 100, 85)
                if st.button("Predict", use_container_width=True):
                    payload = {
                        "mcu_phase": "Phase 4", "release_year": 2024,
                        "tomato_meter": tomato, "audience_score": 85,
                        "movie_duration": 130,
                        "production_budget":   budget  * 1e6,
                        "opening_weekend":     opening * 1e6,
                        "domestic_box_office": opening * 2.5 * 1e6,
                    }
                    resp = requests.post(f"{API_BASE}/predict", json=payload, timeout=5).json()
                    if "predicted_worldwide_box_office_millions" in resp:
                        st.markdown(f'<div class="pred-result">💰 ${resp["predicted_worldwide_box_office_millions"]:.0f}M</div>', unsafe_allow_html=True)
            else:
                st.info("Start the Flask API to enable predictions.")

    # ── DATASET ────────────────────────────────────────────────────────────────
    elif page == "📊 Dataset":
        st.title("📊 MCU Dataset Explorer")
        dataset = get_dataset()
        if not dataset: st.error("API offline."); st.stop()
        df = pd.DataFrame(dataset["data"])
        st.markdown(f"**{dataset['total_entries']} total** · **{dataset['theatrical_count']} theatrical**")
        col1, col2 = st.columns(2)
        sel_phases = col1.multiselect("Filter by Phase", sorted(df["mcu_phase"].dropna().unique()),
                                      default=sorted(df["mcu_phase"].dropna().unique()))
        show_tv    = col2.checkbox("Include Disney+ Shows", value=False)
        mask = df["mcu_phase"].isin(sel_phases)
        if not show_tv: mask &= df["worldwide_box_office"].notna()
        st.dataframe(df[mask][["movie_title","mcu_phase","release_date","tomato_meter",
                                "audience_score","movie_duration","production_budget",
                                "opening_weekend","domestic_box_office","worldwide_box_office"]],
                     use_container_width=True, height=450)
        theatrical = df[df["worldwide_box_office"].notna()].copy()
        if not theatrical.empty:
            st.subheader("Average Worldwide Box Office by Phase")
            phase_avg = theatrical.groupby("mcu_phase")["worldwide_box_office"].mean().sort_index()
            fig, ax = plt.subplots(figsize=(9, 4))
            bars = ax.bar(phase_avg.index, phase_avg.values / 1e6,
                          color=["#3b82d4","#7c5cd8","#10b981","#f59e0b","#ef4444","#6366f1"])
            for b in bars:
                ax.text(b.get_x() + b.get_width()/2, b.get_height() + 5,
                        f"${b.get_height():.0f}M", ha="center", fontsize=9)
            ax.set_ylabel("Avg WW Box Office ($M)")
            ax.set_title("Average Worldwide Box Office by MCU Phase")
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    # ── PREDICT ────────────────────────────────────────────────────────────────
    elif page == "🔮 Predict":
        st.title("🔮 Predict Worldwide Box Office")
        health = get_health()
        if not (health and health.get("status") == "ok"):
            st.error("Flask API is offline. Run: python main.py api"); st.stop()
        with st.form("predict_form"):
            st.subheader("Enter Movie Details")
            col1, col2 = st.columns(2)
            with col1:
                phase        = st.selectbox("MCU Phase", [f"Phase {i}" for i in range(1, 7)], index=3)
                release_year = st.number_input("Release Year", 2008, 2030, 2025)
                tomato       = st.slider("Rotten Tomatoes (%)", 0, 100, 85)
                audience     = st.slider("Audience Score (%)", 0, 100, 88)
            with col2:
                duration = st.number_input("Duration (minutes)", 80, 220, 130)
                budget   = st.number_input("Production Budget ($M)", 50, 600, 200)
                opening  = st.number_input("Opening Weekend ($M)", 5, 500, 100)
                domestic = st.number_input("Domestic Box Office ($M)", 10, 1000, 300)
            submitted = st.form_submit_button("Predict", use_container_width=True)
        if submitted:
            payload = {
                "mcu_phase": phase, "release_year": int(release_year),
                "tomato_meter": tomato, "audience_score": audience,
                "movie_duration": int(duration),
                "production_budget":   budget   * 1e6,
                "opening_weekend":     opening  * 1e6,
                "domestic_box_office": domestic * 1e6,
            }
            try:
                resp = requests.post(f"{API_BASE}/predict", json=payload, timeout=5).json()
                if "error" in resp:
                    st.error(resp["error"])
                else:
                    pred_m = resp["predicted_worldwide_box_office_millions"]
                    pred   = resp["predicted_worldwide_box_office"]
                    st.markdown("---")
                    st.markdown(f'<div class="pred-result">🌍 ${pred_m:,.1f} Million</div>', unsafe_allow_html=True)
                    st.markdown("---")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Predicted WW", f"${pred_m:,.0f}M")
                    c2.metric("Budget", f"${budget:,.0f}M")
                    c3.metric("Est. ROI", f"{(pred - budget*1e6)/(budget*1e6)*100:+.1f}%")
                    comparisons = {
                        "Iron Man (2008)": 585.8, "The Avengers (2012)": 1518.8,
                        "Black Panther (2018)": 1353.3, "Endgame (2019)": 2799.4,
                        "Your Prediction": pred_m,
                    }
                    fig, ax = plt.subplots(figsize=(9, 4))
                    colors = ["#e5e7eb"] * 4 + ["#3b82d4"]
                    ax.barh(list(comparisons.keys()), list(comparisons.values()), color=colors)
                    ax.set_xlabel("Worldwide Box Office ($M)")
                    ax.set_title("Your Prediction vs MCU Milestones")
                    fig.tight_layout(); st.pyplot(fig); plt.close(fig)
            except Exception as e:
                st.error(f"Request failed: {e}")

    # ── ANALYTICS ──────────────────────────────────────────────────────────────
    elif page == "📈 Analytics":
        st.title("📈 Model Analytics")
        preds_data = get_predictions_all()
        feat_data  = get_feature_importance()
        if not preds_data: st.error("API offline."); st.stop()
        df = pd.DataFrame(preds_data["predictions"])

        st.subheader("Actual vs Predicted Worldwide Box Office")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(df["actual_worldwide_box_office"]/1e6, df["predicted_worldwide_box_office"]/1e6,
                   color="#3b82d4", edgecolors="white", s=80, alpha=0.85)
        mn = min(df["actual_worldwide_box_office"].min(), df["predicted_worldwide_box_office"].min())/1e6
        mx = max(df["actual_worldwide_box_office"].max(), df["predicted_worldwide_box_office"].max())/1e6
        ax.plot([mn, mx], [mn, mx], "r--", lw=1.5, label="Perfect prediction")
        ax.set_xlabel("Actual ($M)"); ax.set_ylabel("Predicted ($M)")
        ax.set_title("Actual vs Predicted"); ax.legend()
        for _, row in df.iterrows():
            if abs(row["difference_millions"]) > 200:
                ax.annotate(row["movie_title"],
                            (row["actual_worldwide_box_office"]/1e6, row["predicted_worldwide_box_office"]/1e6),
                            fontsize=7, ha="left", color="#57606a")
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

        if feat_data and "feature_importance" in feat_data:
            st.subheader("Feature Importance")
            imp = feat_data["feature_importance"]
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.barh(list(imp.keys())[::-1], list(imp.values())[::-1], color="#3b82d4")
            ax.set_xlabel("Importance"); ax.set_title("Feature Importance (Best Model)")
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)

        st.subheader("Movie-by-Movie Prediction Details")
        display_df = df.copy()
        display_df["actual_$M"]    = (display_df["actual_worldwide_box_office"]/1e6).round(1)
        display_df["predicted_$M"] = (display_df["predicted_worldwide_box_office"]/1e6).round(1)
        display_df["error_$M"]     = display_df["difference_millions"].round(1)
        st.dataframe(display_df[["movie_title","mcu_phase","release_year",
                                  "actual_$M","predicted_$M","error_$M"]],
                     use_container_width=True, height=420)

        st.subheader("Prediction Errors Distribution")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(df["difference_millions"], bins=12, color="#7c5cd8", edgecolor="white")
        ax.axvline(0, color="red", linestyle="--", lw=1.5)
        ax.set_xlabel("Prediction Error ($M)"); ax.set_ylabel("Count")
        ax.set_title("Distribution of Prediction Errors")
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    # ── MODEL INFO ─────────────────────────────────────────────────────────────
    elif page == "ℹ️ Model Info":
        st.title("ℹ️ Model Information")
        model = get_model_info()
        if not model: st.error("API offline."); st.stop()
        st.subheader(f"Best Model: **{model['best_model']}**")
        m = model["metrics"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("R²",       f"{m['R2']:.4f}")
        c2.metric("MAE",      f"${m['MAE']/1e6:.1f}M")
        c3.metric("RMSE",     f"${m['RMSE']/1e6:.1f}M")
        c4.metric("CV R²",    f"{m['CV_R2_mean']:.4f} ± {m['CV_R2_std']:.4f}")
        st.markdown("---")
        st.subheader("All Model Comparison")
        rows = [{"Model": n, "R²": round(v["R2"],4), "MAE ($M)": round(v["MAE"]/1e6,1),
                 "RMSE ($M)": round(v["RMSE"]/1e6,1),
                 "CV R²": f"{v['CV_R2_mean']:.4f} ± {v['CV_R2_std']:.4f}"}
                for n, v in model["all_models"].items()]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        names  = [r["Model"] for r in rows]
        r2vals = [r["R²"] for r in rows]
        maes   = [r["MAE ($M)"] for r in rows]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].bar(names, r2vals, color=["#3b82d4","#7c5cd8","#10b981"])
        axes[0].set_ylim(0, 1.1); axes[0].set_ylabel("R² Score")
        axes[0].set_title("Model R² Comparison")
        for i, v in enumerate(r2vals): axes[0].text(i, v+0.01, f"{v:.3f}", ha="center")
        axes[1].bar(names, maes, color=["#3b82d4","#7c5cd8","#10b981"])
        axes[1].set_ylabel("MAE ($M)"); axes[1].set_title("Model MAE Comparison")
        for i, v in enumerate(maes): axes[1].text(i, v+1, f"${v:.0f}M", ha="center")
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)
        st.markdown("---")
        st.subheader("Features Used")
        for i, feat in enumerate(model["features"], 1):
            st.markdown(f"**{i}.** `{feat}`")
        st.markdown("---")
        st.subheader("Raw Metadata (JSON)")
        st.json(model)


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
# When Streamlit imports this file it passes `--ui-mode` as a script arg,
# so we detect that and render the app instead of running CLI dispatch.
if "--ui-mode" in sys.argv or (len(sys.argv) >= 1 and "streamlit" in sys.argv[0].lower()):
    _streamlit_app()

elif __name__ == "__main__":
    commands = {"train", "api", "ui", "all"}
    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        print(__doc__)
        sys.exit(0)

    cmd = sys.argv[1]

    if cmd == "train":
        run_training()

    elif cmd == "api":
        run_api()

    elif cmd == "ui":
        run_ui()

    elif cmd == "all":
        print("Step 1/2 – Training model...\n")
        run_training()
        print("\nStep 2/2 – Starting API...\n")
        run_api()
