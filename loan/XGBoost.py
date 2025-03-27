from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    RocCurveDisplay, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import pandas as pd


class XGBoostLoanModel:
    def __init__(self, params=None):
        self.model = XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            scale_pos_weight=1,
            random_state=42,
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            **(params or {})
        )
        self.cleaned_columns = None  # To track original feature names

    def _sanitize_column_names(self, df):
        cleaned_df = df.copy()
        cleaned_df.columns = cleaned_df.columns.str.replace(r"[<>[\]]", "_", regex=True)
        self.cleaned_columns = cleaned_df.columns
        return cleaned_df

    def fit(self, X_train, y_train):
        X_train_clean = self._sanitize_column_names(X_train)
        self.model.fit(X_train_clean, y_train)

    def predict(self, X_test):
        X_test_clean = self._sanitize_column_names(X_test)
        return self.model.predict(X_test_clean)

    def predict_proba(self, X_test):
        X_test_clean = self._sanitize_column_names(X_test)
        return self.model.predict_proba(X_test_clean)[:, 1]

    def evaluate(self, X_test, y_test, return_metrics=False):
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
        RocCurveDisplay.from_estimator(self.model, self._sanitize_column_names(X_test), y_test)
        plt.title("ROC Curve")
        plt.grid(True)
        plt.show()

        ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
        plt.title("Confusion Matrix")
        plt.grid(False)
        plt.show()

        

    def feature_importance(self, X_train):
        X_train_clean = self._sanitize_column_names(X_train)
        importance_df = pd.DataFrame({
            'Feature': X_train_clean.columns,
            'Importance': self.model.feature_importances_
        }).sort_values(by='Importance', ascending=False)

        print("\n🔍 Top Feature Importances:")
        print(importance_df.head(20))

        importance_df.head(20).plot(
            kind='barh', x='Feature', y='Importance',
            title='Top 20 Feature Importances (XGBoost)', figsize=(8, 6)
        )
        plt.gca().invert_yaxis()
        plt.grid(True)
        plt.show()

        return importance_df
