"""
ML Capstone Dashboard - Streamlit web app for the Week 4 capstone projects.
Run with:  streamlit run app.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import ml_core as mc

st.set_page_config(page_title="ML Capstone Dashboard", page_icon="📊", layout="wide")

MODEL_DIR = Path("saved_models")


# ------------------------------------------------------------ plot helpers
def confusion_fig(cm):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1], labels=["Actual 0", "Actual 1"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title("Confusion matrix")
    return fig


def roc_fig(fpr, tpr, auc):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Random guess")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve")
    ax.legend(loc="lower right")
    return fig


# ----------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("1. Choose a project")
    project = st.selectbox(
        "Project",
        list(mc.PROJECTS),
        index=1,  # 0 = Heart Disease, 1 = Bank Loan Approval, 2 = Mall Customers
        format_func=lambda k: mc.PROJECTS[k]["title"],
    )
    cfg = mc.PROJECTS[project]
    st.caption(f"Dataset: {cfg['dataset']}")

    st.header("2. Choose data")
    source = st.radio("Source", ["Built-in synthetic sample", "Upload Kaggle CSV"])
    if source == "Upload Kaggle CSV":
        uploaded = st.file_uploader("CSV file", type="csv")
        if uploaded is None:
            st.info("Upload the CSV to continue, or switch to the built-in sample.")
            st.stop()
        df = pd.read_csv(uploaded)
    else:
        df = mc.make_sample(project)
        st.caption("Synthetic demo data with the same columns as the Kaggle file. "
                   "Upload the real CSV for your final results.")

st.title("ML Capstone Dashboard")
st.caption(f"{cfg['title']} · full pipeline: explore → train → evaluate → tune → save → predict")

same_result = lambda key: st.session_state.get("result_key") == key  # noqa: E731

# --------------------------------------------------------- classification
if cfg["task"] == "classification":
    tab_explore, tab_train, tab_tune, tab_save = st.tabs(
        ["Explore", "Train & evaluate", "Cross-validation & tuning", "Save & predict"]
    )

    with tab_explore:
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", df.shape[0])
        c2.metric("Columns", df.shape[1])
        c3.metric("Missing cells", int(df.isna().sum().sum()))
        st.dataframe(df.head(20))
        st.subheader("Summary statistics")
        st.dataframe(df.describe(include="all").T)

    with tab_train:
        model_name = st.selectbox("Model", list(mc.CLASSIFIERS))
        c1, c2 = st.columns(2)
        test_size = c1.slider("Test split", 0.1, 0.4, 0.2, 0.05)
        cv_folds = c2.slider("Cross-validation folds", 3, 10, 5)

        if st.button("Train model", type="primary"):
            with st.spinner("Training..."):
                try:
                    st.session_state["result"] = mc.train_classifier(
                        df, project, model_name, test_size, cv_folds
                    )
                    st.session_state["result_key"] = project
                except ValueError as err:
                    st.error(str(err))

        res = st.session_state.get("result")
        if res is None or not same_result(project):
            st.info("Click **Train model** to see the results.")
        else:
            cols = st.columns(5)
            for col, (name, value) in zip(cols, res.metrics.items()):
                col.metric(name.replace("_", " ").title(), f"{value:.3f}")
            st.write(f"**Cross-validation accuracy:** {res.cv_scores.mean():.3f} "
                     f"± {res.cv_scores.std():.3f} across {len(res.cv_scores)} folds")
            c1, c2 = st.columns(2)
            c1.pyplot(confusion_fig(res.confusion))
            c2.pyplot(roc_fig(res.fpr, res.tpr, res.metrics["roc_auc"]))
            with st.expander("Full classification report"):
                st.code(res.report)

    with tab_tune:
        st.write(
            "Runs **GridSearchCV** on an SVM pipeline (scaling + encoding + SVC), trying "
            "every combination of C, kernel and gamma with cross-validation. Tuning uses "
            "only the training split, so the test accuracy below is an honest score."
        )
        if st.button("Run GridSearchCV for SVM"):
            with st.spinner("Searching parameter grid..."):
                search, test_acc = mc.tune_svm(df, project)
                st.session_state["grid"] = (search, test_acc, project)

        if "grid" in st.session_state and st.session_state["grid"][2] == project:
            search, test_acc, _ = st.session_state["grid"]
            c1, c2 = st.columns(2)
            c1.metric("Best cross-validation accuracy", f"{search.best_score_:.3f}")
            c2.metric("Held-out test accuracy", f"{test_acc:.3f}")
            st.write("**Best parameters:**", search.best_params_)
            table = pd.DataFrame(search.cv_results_)[
                ["params", "mean_test_score", "std_test_score", "rank_test_score"]
            ].sort_values("rank_test_score").head(10)
            table["params"] = table["params"].astype(str)
            st.dataframe(table)

    with tab_save:
        res = st.session_state.get("result")
        st.subheader("Save the trained model")
        if res is None or not same_result(project):
            st.info("Train a model in the **Train & evaluate** tab first.")
        elif st.button("Save trained model to disk"):
            fname = f"{project}_{res.model_name.lower().replace(' ', '_')}.joblib"
            path = mc.save_model(res.pipeline, MODEL_DIR / fname)
            st.success(f"Saved to `{path}`")

        st.subheader("Load a saved model and predict on new data")
        saved = sorted(MODEL_DIR.glob(f"{project}_*.joblib")) if MODEL_DIR.exists() else []
        if not saved:
            st.info("No saved models for this project yet.")
        else:
            choice = st.selectbox("Saved model", saved, format_func=lambda p: p.name)
            new_file = st.file_uploader(
                "New rows to predict (CSV with the same feature columns)", type="csv"
            )
            if new_file is not None and st.button("Predict"):
                model = mc.load_model(choice)
                preds = mc.predict_new(model, pd.read_csv(new_file), project)
                st.dataframe(preds)
                st.download_button("Download predictions",
                                   preds.to_csv(index=False), "predictions.csv", "text/csv")

# ------------------------------------------------------------- clustering
else:
    tab_explore, tab_cluster = st.tabs(["Explore", "Customer segments"])

    with tab_explore:
        st.metric("Customers", df.shape[0])
        st.dataframe(df.head(20))
        st.dataframe(df.describe(include="all").T)

    with tab_cluster:
        features = [c for c in cfg["features"] if c in df.columns]
        if len(features) < len(cfg["features"]):
            st.error(f"This project needs the columns {cfg['features']}. "
                     f"Your file has: {list(df.columns)}")
        else:
            k = st.slider("Number of segments (k)", 2, 8, 3)
            ks, inertias = mc.elbow_inertias(df, features)
            seg, _ = mc.segment_customers(df, features, k)

            c1, c2 = st.columns(2)
            with c1:
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.plot(ks, inertias, "o-")
                ax.set_xlabel("k (number of clusters)")
                ax.set_ylabel("Inertia")
                ax.set_title("Elbow method: pick k where the curve bends")
                st.pyplot(fig)
            with c2:
                fig, ax = plt.subplots(figsize=(5, 4))
                for name, group in seg.groupby("segment"):
                    ax.scatter(group["Annual Income (k$)"],
                               group[cfg["spending_col"]], label=name, alpha=0.7)
                ax.set_xlabel("Annual Income (k$)")
                ax.set_ylabel("Spending Score (1-100)")
                ax.set_title("Customer segments")
                ax.legend()
                st.pyplot(fig)

            summary = seg.groupby("segment")[features].mean().round(1)
            summary["customers"] = seg.groupby("segment").size()
            st.subheader("Segment summary (average values)")
            st.dataframe(summary)
            st.download_button("Download segmented customers",
                               seg.to_csv(), "customer_segments.csv", "text/csv")
