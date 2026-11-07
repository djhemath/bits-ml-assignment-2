"""Spambase Model Lab - a Streamlit front end for five pre-trained spam classifiers.

The app loads the saved scikit-learn pipelines from ``model/`` and evaluates them
on an uploaded test CSV. Nothing is trained or re-fitted here.
"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
SAMPLE_CSV = BASE_DIR / "test_data.csv"
TARGET = "is_spam"
CLASS_NAMES = ["Not spam", "Spam"]

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "SVM": "svm.joblib",
}
METRIC_ORDER = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]

st.set_page_config(page_title="Spambase Model Lab", layout="wide")


# --------------------------------------------------------------------------- #
# Loading and scoring
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading saved models...")
def load_models():
    """Load every saved pipeline once per server process."""
    return {name: joblib.load(MODEL_DIR / fname) for name, fname in MODEL_FILES.items()}


def required_features(models):
    """The feature columns (and their order) the saved models were trained on."""
    return list(next(iter(models.values())).feature_names_in_)


def validate_upload(df, features):
    """Return (clean_df, list_of_problems). Problems make the upload unusable."""
    problems = []
    missing = [c for c in features + [TARGET] if c not in df.columns]
    if missing:
        shown = ", ".join(missing[:8]) + (" ..." if len(missing) > 8 else "")
        problems.append(f"Missing {len(missing)} required column(s): {shown}")
        return None, problems

    clean = df[features + [TARGET]].copy()
    non_numeric = [c for c in clean.columns if not pd.api.types.is_numeric_dtype(clean[c])]
    if non_numeric:
        problems.append(f"Non-numeric column(s): {', '.join(non_numeric[:8])}")
    if clean.isna().any().any():
        n_bad = int(clean.isna().any(axis=1).sum())
        problems.append(f"{n_bad} row(s) contain missing values.")
    if problems:
        return None, problems

    if not set(clean[TARGET].unique()) <= {0, 1}:
        problems.append(f"`{TARGET}` must contain only 0 and 1.")
    if (clean[features] < 0).any().any():
        problems.append("Feature values must be non-negative (the models apply log(1+x)).")
    if problems:
        return None, problems
    return clean, []


def score_models(models, X, y, threshold):
    """Predict with every saved model and collect probabilities and metrics."""
    probs, rows = {}, {}
    for name, model in models.items():
        p = model.predict_proba(X)[:, 1]
        pred = (p >= threshold).astype(int)
        probs[name] = p
        rows[name] = {
            "Accuracy": accuracy_score(y, pred),
            "AUC": roc_auc_score(y, p) if y.nunique() == 2 else np.nan,
            "Precision": precision_score(y, pred, zero_division=0),
            "Recall": recall_score(y, pred, zero_division=0),
            "F1": f1_score(y, pred, zero_division=0),
            "MCC": matthews_corrcoef(y, pred),
        }
    table = pd.DataFrame(rows).T[METRIC_ORDER]
    table.index.name = "Model"
    return probs, table


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #
st.title("Spambase Model Lab")
st.caption(
    "Upload test e-mails described by the 57 Spambase features, pick a saved model "
    "and see how it performs. The models are pre-trained and are not re-fitted on your file."
)

try:
    models = load_models()
except FileNotFoundError as exc:
    st.error(f"Saved model file not found: {exc.filename}. Run the notebook to create `model/`.")
    st.stop()
features = required_features(models)

with st.sidebar:
    st.header("1 · Test data")
    uploaded = st.file_uploader("Upload a test CSV", type="csv")
    if SAMPLE_CSV.exists():
        st.download_button(
            "Download sample test_data.csv",
            data=SAMPLE_CSV.read_bytes(),
            file_name="test_data.csv",
            mime="text/csv",
            help="The held-out test split from the notebook. Upload it back to try the app.",
        )
    with st.expander("Required CSV format"):
        st.write(
            f"One row per e-mail. The file needs the {len(features)} feature columns "
            f"(e.g. `{features[0]}`, `{features[-1]}`) plus a `{TARGET}` column with "
            "1 = spam and 0 = not spam. Extra columns are ignored."
        )

    st.header("2 · Model")
    chosen = st.selectbox("Choose a classifier", list(MODEL_FILES))

    st.header("3 · Decision threshold")
    threshold = st.slider(
        "Classify as spam if P(spam) ≥",
        0.05, 0.95, 0.50, 0.05,
        help="Changes precision, recall, F1, MCC and the confusion matrix. "
             "AUC does not depend on the threshold.",
    )

if uploaded is None:
    st.info("Upload a test CSV in the sidebar to begin. You can download a ready-made sample there.")
    st.stop()

try:
    uploaded.seek(0)  # the buffer may already have been read on an earlier rerun
    raw_df = pd.read_csv(uploaded)
except Exception as exc:  # unreadable or empty file
    st.error(f"Could not read the file as CSV: {exc}")
    st.stop()

data, problems = validate_upload(raw_df, features)
if problems:
    st.error("The uploaded file cannot be used:")
    for msg in problems:
        st.markdown(f"- {msg}")
    st.stop()

X, y = data[features], data[TARGET].astype(int)
extra_cols = [c for c in raw_df.columns if c not in features + [TARGET]]

n_spam = int(y.sum())
c1, c2, c3 = st.columns(3)
c1.metric("E-mails in file", f"{len(y):,}")
c2.metric("Spam", f"{n_spam:,} ({n_spam / len(y):.1%})")
c3.metric("Not spam", f"{len(y) - n_spam:,} ({1 - n_spam / len(y):.1%})")
if extra_cols:
    st.caption(f"Ignored {len(extra_cols)} extra column(s): {', '.join(extra_cols[:5])}"
               + (" ..." if len(extra_cols) > 5 else ""))
if y.nunique() < 2:
    st.warning("The file contains only one class, so AUC cannot be computed and MCC will be 0.")

probs, table = score_models(models, X, y, threshold)
chosen_pred = (probs[chosen] >= threshold).astype(int)

tab_metrics, tab_cm, tab_compare = st.tabs(
    ["Metrics", "Confusion matrix & report", "Compare all models"]
)

with tab_metrics:
    st.subheader(f"{chosen}: evaluation metrics")
    st.caption(f"Threshold {threshold:.2f} · positive class = spam")
    # Two rows of three so values are never truncated on narrower windows
    for row in (METRIC_ORDER[:3], METRIC_ORDER[3:]):
        for col, metric in zip(st.columns(3), row):
            value = table.loc[chosen, metric]
            col.metric(metric, "n/a" if pd.isna(value) else f"{value:.4f}")
    st.markdown(
        "- **Precision**: of the e-mails flagged as spam, the share that really are spam.\n"
        "- **Recall**: of all real spam, the share that was caught.\n"
        "- **MCC**: a single balanced score from −1 to 1 that stays honest when classes are unequal."
    )

with tab_cm:
    left, right = st.columns([1, 1.2])
    with left:
        st.subheader("Confusion matrix")
        fig, ax = plt.subplots(figsize=(4.4, 4))
        ConfusionMatrixDisplay(
            confusion_matrix(y, chosen_pred, labels=[0, 1]), display_labels=CLASS_NAMES
        ).plot(ax=ax, cmap="Blues", colorbar=False)
        ax.grid(False)
        st.pyplot(fig)
        plt.close(fig)
        tn, fp, fn, tp = confusion_matrix(y, chosen_pred, labels=[0, 1]).ravel()
        st.caption(
            f"{fp} legitimate e-mail(s) wrongly flagged as spam · {fn} spam e-mail(s) missed."
        )
    with right:
        st.subheader("Classification report")
        report = classification_report(
            y, chosen_pred, labels=[0, 1], target_names=CLASS_NAMES,
            output_dict=True, zero_division=0,
        )
        report_df = pd.DataFrame(report).T
        report_df.loc["accuracy", "support"] = len(y)  # sklearn stores only a scalar here
        report_df["support"] = report_df["support"].astype(int)
        st.dataframe(
            report_df.style.format({"precision": "{:.3f}", "recall": "{:.3f}", "f1-score": "{:.3f}"}),
        )

with tab_compare:
    st.subheader("All five models on your file")
    st.caption(f"Same threshold ({threshold:.2f}) for every model. Best value in each column is highlighted.")
    st.dataframe(
        table.style.format("{:.4f}", na_rep="n/a").highlight_max(
            axis=0, props="background-color: #1b5e20; color: #ffffff; font-weight: 700;"
        ),
    )

    g1, g2 = st.columns(2)
    with g1:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        table.plot.bar(ax=ax, width=0.82, colormap="tab10", edgecolor="white")
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("")
        ax.set_title("Metrics by model")
        ax.legend(ncol=3, fontsize=8, loc="lower center")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    with g2:
        if y.nunique() == 2:
            fig, ax = plt.subplots(figsize=(6.5, 4))
            for name, p in probs.items():
                fpr, tpr, _ = roc_curve(y, p)
                ax.plot(fpr, tpr, lw=2, label=f"{name} ({table.loc[name, 'AUC']:.3f})")
            ax.plot([0, 1], [0, 1], "k--", lw=1)
            ax.set_xlabel("False positive rate")
            ax.set_ylabel("True positive rate")
            ax.set_title("ROC curves (AUC in legend)")
            ax.legend(loc="lower right", fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("ROC curves need both classes in the uploaded file.")

    with st.expander("Per-e-mail predictions for the selected model"):
        out = pd.DataFrame({
            "P(spam)": probs[chosen].round(4),
            "predicted": np.where(chosen_pred == 1, "Spam", "Not spam"),
            "actual": np.where(y.values == 1, "Spam", "Not spam"),
        })
        out["correct"] = out["predicted"] == out["actual"]
        st.dataframe(out, height=300)
        st.download_button(
            "Download predictions (CSV)",
            out.to_csv(index=False).encode(),
            file_name=f"predictions_{chosen.lower().replace(' ', '_')}.csv",
            mime="text/csv",
        )
