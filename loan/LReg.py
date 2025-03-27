from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    RocCurveDisplay, ConfusionMatrixDisplay
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd


class LogisticRegressionLoanModel:
    def __init__(self, scale_data=True, params=None):
        self.scale_data = scale_data
        self.scaler = StandardScaler() if scale_data else None
        self.model = LogisticRegression(
            class_weight='balanced',
            solver='liblinear',
            max_iter=200,
            random_state=42,
            **(params or {})
        )

    def fit(self, X_train, y_train):
        if self.scale_data:
            X_train = self.scaler.fit_transform(X_train)
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        if self.scale_data:
            X_test = self.scaler.transform(X_test)
        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        if self.scale_data:
            X_test = self.scaler.transform(X_test)
        return self.model.predict_proba(X_test)[:, 1]

    def evaluate(self, X_test, y_test, return_metrics=False):
        if self.scale_data:
            X_test_scaled = self.scaler.transform(X_test)
        else:
            X_test_scaled = X_test
        
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)

        # 计算分类报告（包含 accuracy）
        report = classification_report(y_test, y_pred, output_dict=True)
        auc = roc_auc_score(y_test, y_pred_proba)
        cm = confusion_matrix(y_test, y_pred)

        # 提取 accuracy
        accuracy = report['accuracy']
        if return_metrics:
            return {
                'accuracy': accuracy,
                'classification_report': report,
                'auc': auc,
                'confusion_matrix': cm
            }
        print("\n📊 Classification Report:")
        print(classification_report(y_test, y_pred))
        print("🎯 AUC-ROC:", auc)
        print("🧩 Confusion Matrix:")
        print(cm)
        print(f"✅ Accuracy: {accuracy:.4f}")

        # 可视化 ROC 曲线和混淆矩阵
        RocCurveDisplay.from_estimator(self.model, X_test_scaled, y_test)
        plt.title("ROC Curve")
        plt.grid(True)
        plt.show()

        ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
        plt.title("Confusion Matrix")
        plt.grid(False)
        plt.show()

        

    def feature_importance(self, X_train):
        coef = self.model.coef_[0]
        feature_names = X_train.columns if hasattr(X_train, 'columns') else [f'x{i}' for i in range(len(coef))]
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': coef
        }).sort_values(by='Importance', key=abs, ascending=False)

        print("\n🔍 Top Feature Coefficients (by abs):")
        print(importance_df.head(20))

        importance_df.head(20).plot(
            kind='barh', x='Feature', y='Importance',
            title='Top 20 Feature Coefficients (Logistic Regression)', figsize=(8, 6)
        )
        plt.gca().invert_yaxis()
        plt.grid(True)
        plt.show()

        return importance_df
