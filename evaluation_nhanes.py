import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from preparedata_nhanes import load_nhanes_patient_graph
from gcn_model import GCN

data_dir = "./data"

def evaluate():
    # Load data
    X, y, train_idx, val_idx, test_idx, adj = load_nhanes_patient_graph(data_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X = X.to(device)
    y = y.to(device)
    adj = adj.to(device)
    test_idx = test_idx.to(device)

    # Load the trained model
    model = GCN(
        in_feats=X.shape[1],
        h_feats=64,
        num_classes=2
    ).to(device)

    model.load_state_dict(torch.load("best_gcn_model.pth", map_location=device))
    model.eval()

    # Predict
    with torch.no_grad():
        out = model(X, adj)
        preds = out.argmax(dim=1)

    # Evaluation
    print("\n========== CLASSIFICATION REPORT ==========")
    print(classification_report(y[test_idx].cpu(), preds[test_idx].cpu()))

    print("========== CONFUSION MATRIX ==========")
    print(confusion_matrix(y[test_idx].cpu(), preds[test_idx].cpu()))

    # ROC-AUC
    probs = F.softmax(out, dim=1)[:, 1].detach().cpu().numpy()
    auc = roc_auc_score(y.cpu().numpy(), probs)

    print("========== ROC-AUC ==========")
    print("ROC-AUC:", auc)

if __name__ == "__main__":
    evaluate()
