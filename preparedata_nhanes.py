import os
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors


###############################################################
# SAFE CSV READER
###############################################################
def read_csv_safe(path):
    try:
        return pd.read_csv(path, encoding="utf-8", engine="python")
    except:
        return pd.read_csv(path, encoding="latin-1", engine="python")


###############################################################
# MAIN FUNCTION
###############################################################
def load_nhanes_patient_graph(
    data_dir: str,
    label_col: str = "DIQ010",
    k_neighbors: int = 10,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
):
    """
    Build a patient-centric graph from NHANES:
    - Node: patients (SEQN)
    - Features: merged numeric columns from demographic, diet, exam, labs, meds, questionnaire
    - Label: DIQ010 (1=diabetes, 2=non-diabetes) -> binary
    - Graph: k-NN (patient-patient)

    Return:
        X, y, train_idx, val_idx, test_idx, adj
    """

    ###########################################################
    # LOAD 6 NHANES TABLES
    ###########################################################
    demo = read_csv_safe(os.path.join(data_dir, "demographic.csv"))
    diet = read_csv_safe(os.path.join(data_dir, "diet.csv"))
    exam = read_csv_safe(os.path.join(data_dir, "examination.csv"))
    labs = read_csv_safe(os.path.join(data_dir, "labs.csv"))
    meds = read_csv_safe(os.path.join(data_dir, "medications.csv"))
    ques = read_csv_safe(os.path.join(data_dir, "questionnaire.csv"))


    ###########################################################
    # LABEL: DIABETES
    ###########################################################
    if label_col not in ques.columns:
        raise ValueError(f"Label column '{label_col}' not found in questionnaire.csv")

    ques = ques[["SEQN", label_col]]
    ques = ques[ques[label_col].isin([1, 2])]  # keep only yes/no
    ques["label"] = (ques[label_col] == 1).astype(int)

    df = ques[["SEQN", "label"]].copy()


    ###########################################################
    # MERGE FEATURES
    ###########################################################
    def merge(df, other, suffix):
        other = other.rename(
            columns={c: f"{c}_{suffix}" for c in other.columns if c != "SEQN"}
        )
        return df.merge(other, on="SEQN", how="left")

    df = merge(df, demo, "demo")
    df = merge(df, diet, "diet")
    df = merge(df, exam, "exam")

    ###########################################################
    # LAB FEATURES (wide numeric)
    ###########################################################
    lab_numeric_cols = [
        c for c in labs.columns if c != "SEQN" and pd.api.types.is_numeric_dtype(labs[c])
    ]
    labs_num = labs[["SEQN"] + lab_numeric_cols]
    df = df.merge(labs_num, on="SEQN", how="left")

    ###########################################################
    # MEDICATION FEATURES (long → wide)
    ###########################################################
    if "RXDDRUG" in meds.columns and "RXDUSE" in meds.columns:
        meds_wide = meds.pivot_table(
            index="SEQN",
            columns="RXDDRUG",
            values="RXDUSE",
            aggfunc="max"
        ).fillna(0)
        meds_wide = meds_wide.reset_index()
    else:
        meds_wide = pd.DataFrame({"SEQN": df["SEQN"], "NO_MED": 0})

    df = df.merge(meds_wide, on="SEQN", how="left")


    ###########################################################
    # BUILD FEATURES
    ###########################################################
    feature_cols = [
        c for c in df.columns
        if c not in ["SEQN", "label"] and pd.api.types.is_numeric_dtype(df[c])
    ]

    if len(feature_cols) == 0:
        raise RuntimeError("No numeric features found in merged NHANES dataset!")

    features = df[feature_cols].fillna(0.0).astype(np.float32)

    # simple normalization (z-score)
    mean = features.mean()
    std = features.std().replace(0, 1.0)
    features = (features - mean) / std

    X_np = features.to_numpy(dtype=np.float32)
    y_np = df["label"].to_numpy(dtype=np.int64)

    N = X_np.shape[0]
    indices = np.arange(N)


    ###########################################################
    # SPLIT DATA
    ###########################################################
    idx_train_val, idx_test = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=y_np,
    )

    val_relative = val_size / (1 - test_size)

    idx_train, idx_val = train_test_split(
        idx_train_val,
        test_size=val_relative,
        random_state=random_state,
        stratify=y_np[idx_train_val],
    )


    ###########################################################
    # BUILD k-NN GRAPH
    ###########################################################
    k = min(k_neighbors, max(1, N - 1))

    nbrs = NearestNeighbors(
        n_neighbors=k + 1,
        metric="euclidean",
        n_jobs=-1
    ).fit(X_np)

    distances, knn_indices = nbrs.kneighbors(X_np)

    # remove self-loop at column 0
    knn_indices = knn_indices[:, 1:]  # shape = [N, k]

    rows = np.repeat(np.arange(N), k)
    cols = knn_indices.reshape(-1)
    values = np.ones(len(rows), dtype=np.float32)

    coords = np.vstack([rows, cols])
    coords_t = torch.tensor(coords, dtype=torch.long)
    values_t = torch.tensor(values, dtype=torch.float32)

    adj = torch.sparse_coo_tensor(
        coords_t,
        values_t,
        size=(N, N)
    ).coalesce()

    # symmetrize
    adj_T = torch.sparse_coo_tensor(
        torch.stack([coords_t[1], coords_t[0]]),
        values_t,
        size=(N, N)
    )

    adj = (adj + adj_T).coalesce()
    mask_vals = torch.where(adj.values() > 0, torch.ones_like(adj.values()), adj.values())
    adj = torch.sparse_coo_tensor(adj.indices(), mask_vals, adj.size()).coalesce()


    ###########################################################
    # RETURN
    ###########################################################
    X = torch.from_numpy(X_np)
    y = torch.from_numpy(y_np)
    train_idx = torch.from_numpy(idx_train)
    val_idx = torch.from_numpy(idx_val)
    test_idx = torch.from_numpy(idx_test)

    return X, y, train_idx, val_idx, test_idx, adj
