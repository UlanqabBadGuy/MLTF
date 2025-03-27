from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    RocCurveDisplay, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import pandas as pd


class CatBoostLoanModel:
    def __init__(self, categorical_features=None, params=None, verbose=0):
        self.categorical_features = categorical_features or []

        default_params = dict(
            iterations=200,
            learning_rate=0.05,
            depth=6,
            random_state=42,
            auto_class_weights='Balanced',
            verbose=verbose
        )
        default_params.update(params or {})

        self.model = CatBoostClassifier(**default_params)


    def fit(self, X_train, y_train):
        self.model.fit(
            X_train,
            y_train,
            cat_features=self.categorical_features
        )

    def predict(self, X_test):
        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test)[:, 1]

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
        RocCurveDisplay.from_estimator(self.model, X_test, y_test)
        plt.title("ROC Curve")
        plt.grid(True)
        plt.show()

        ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
        plt.title("Confusion Matrix")
        plt.grid(False)
        plt.show()

       

    def feature_importance(self, X_train):
        importance_df = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': self.model.get_feature_importance()
        }).sort_values(by='Importance', ascending=False)

        print("\n🔍 Top Feature Importances (CatBoost):")
        print(importance_df.head(20))

        importance_df.head(20).plot(
            kind='barh', x='Feature', y='Importance',
            title='Top 20 Feature Importances (CatBoost)', figsize=(8, 6)
        )
        plt.gca().invert_yaxis()
        plt.grid(True)
        plt.show()

        return importance_df
