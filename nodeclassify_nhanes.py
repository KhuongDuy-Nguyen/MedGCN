from preparedata_nhanes import load_nhanes_patient_graph

data_dir = "../data"

X, y, train_idx, val_idx, test_idx, adj = load_nhanes_patient_graph(data_dir)

print("X shape:", X.shape)
print("y shape:", y.shape)
print("train/val/test:", len(train_idx), len(val_idx), len(test_idx))
print("adj:", adj.shape, " num edges:", adj._nnz())

