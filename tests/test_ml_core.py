import numpy as np
import pytest

import ml_core as mc


@pytest.mark.parametrize("project", ["heart", "loan"])
@pytest.mark.parametrize("model_name", list(mc.CLASSIFIERS))
def test_every_classifier_trains_and_reports_metrics(project, model_name):
    df = mc.make_sample(project, n=200)
    res = mc.train_classifier(df, project, model_name, cv_folds=3)
    assert set(res.metrics) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert all(0 <= v <= 1 for v in res.metrics.values())
    assert len(res.cv_scores) == 3
    assert res.confusion.shape == (2, 2)


def test_heart_target_built_from_num_column():
    df = mc.make_sample("heart", n=50).drop(columns=["target"])
    df["num"] = np.where(np.arange(50) % 2 == 0, 0, 2)
    X, y = mc.prepare_xy(df, "heart")
    assert set(y.unique()) == {0, 1}
    assert "num" not in X.columns


def test_loan_status_mapped_to_binary_and_id_dropped():
    X, y = mc.prepare_xy(mc.make_sample("loan", n=100), "loan")
    assert set(y.unique()) <= {0, 1}
    assert "Loan_ID" not in X.columns
    assert "Loan_Status" not in X.columns


def test_missing_target_column_gives_clear_error():
    with pytest.raises(ValueError, match="Target column"):
        mc.prepare_xy(mc.make_sample("heart", n=20).drop(columns=["target"]), "heart")


def test_save_load_predict_roundtrip(tmp_path):
    res = mc.train_classifier(mc.make_sample("loan", n=200), "loan", "Decision Tree", cv_folds=3)
    path = mc.save_model(res.pipeline, tmp_path / "loan_tree.joblib")
    model = mc.load_model(path)
    out = mc.predict_new(model, mc.make_sample("loan", n=5, seed=7), "loan")
    assert len(out) == 5
    assert set(out["prediction_label"]) <= {"Approved", "Not approved"}
    assert out["probability"].between(0, 1).all()


def test_tune_svm_returns_best_params():
    search, test_acc = mc.tune_svm(mc.make_sample("heart", n=200), "heart", cv_folds=3)
    assert set(search.best_params_) == set(mc.SVM_GRID)
    assert 0 <= test_acc <= 1


def test_segment_customers_names_by_spending():
    df = mc.make_sample("mall", n=300)
    seg, _ = mc.segment_customers(df, mc.PROJECTS["mall"]["features"], k=3)
    assert set(seg["segment"]) == {"Low Spenders", "Medium Spenders", "High Spenders"}
    means = seg.groupby("segment")["Spending Score (1-100)"].mean()
    assert means["Low Spenders"] < means["High Spenders"]


def test_elbow_inertia_never_increases():
    df = mc.make_sample("mall", n=200)
    _, inertias = mc.elbow_inertias(df, mc.PROJECTS["mall"]["features"], max_k=5)
    assert all(a >= b for a, b in zip(inertias, inertias[1:]))
