# Re-import libraries due to code state reset
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


class LoanDefaultMLP(nn.Module):
    def __init__(self, input_dim):
        super(LoanDefaultMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)


class MLPTrainer:
    def __init__(self, X_train, y_train, X_test, y_test, batch_size=64, lr=1e-3, epochs=15):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = StandardScaler()
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs

        # 数据标准化
        self.X_train = self.scaler.fit_transform(X_train)
        self.X_test = self.scaler.transform(X_test)

        self.y_train = np.array(y_train).reshape(-1).astype(int)
        self.y_test = np.array(y_test).reshape(-1).astype(int)

        self.train_loader = self._create_loader(self.X_train, self.y_train, shuffle=True)
        self.test_loader = self._create_loader(self.X_test, self.y_test, shuffle=False)

        self.model = LoanDefaultMLP(self.X_train.shape[1]).to(self.device)
        self.criterion = self._get_loss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)



    def _create_loader(self, X, y, shuffle):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    # def _get_loss(self):
    #     class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(self.y_train), y=self.y_train)
    #     pos_weight = torch.tensor(class_weights[1], dtype=torch.float32).to(self.device)
    #     return nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # 损失函数：改成自动设置比例
    def _get_loss(self):
        class_counts = np.bincount(self.y_train)
        pos_weight = torch.tensor([class_counts[0] / class_counts[1]], dtype=torch.float32).to(self.device)
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0
            for X_batch, y_batch in self.train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                self.optimizer.zero_grad()
                logits = self.model(X_batch).squeeze()
                loss = self.criterion(logits, y_batch)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            print(f"Epoch {epoch + 1}/{self.epochs} - Loss: {epoch_loss:.4f}")

    def evaluate(self):
        self.model.eval()
        all_probs, all_preds, all_trues = [], [], []

        with torch.no_grad():
            for X_batch, y_batch in self.test_loader:
                X_batch = X_batch.to(self.device)
                logits = self.model(X_batch).squeeze()
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).int().cpu().numpy()

                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds)
                all_trues.extend(y_batch.numpy())

        print("\n📊 Classification Report:")
        print(classification_report(all_trues, all_preds))
        print("🎯 AUC-ROC:", roc_auc_score(all_trues, all_probs))
        print("🧩 Confusion Matrix:")
        print(confusion_matrix(all_trues, all_preds))
