# Re-import necessary libraries due to code execution state reset
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    RocCurveDisplay, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import pandas as pd


class LightGBMLoanModel:
    def __init__(self, categorical_features=None, params=None):
        self.categorical_features = categorical_features or []
        self.model = LGBMClassifier(
            is_unbalance=True,
            random_state=42,
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=64,
            max_depth=-1,
            **(params or {})
        )

    def fit(self, X_train, y_train):
        for col in self.categorical_features:
            X_train[col] = X_train[col].astype('category')
        self.model.fit(X_train, y_train, categorical_feature=self.categorical_features)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test)[:, 1]

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)

        print("\n📊 Classification Report:")
        print(classification_report(y_test, y_pred))
        print("🎯 AUC-ROC:", roc_auc_score(y_test, y_pred_proba))
        print("🧩 Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

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
            'Importance': self.model.feature_importances_
        }).sort_values(by='Importance', ascending=False)

        print("\n🔍 Top Feature Importances:")
        print(importance_df.head(20))

        importance_df.head(20).plot(
            kind='barh', x='Feature', y='Importance',
            title='Top 20 Feature Importances', figsize=(8, 6)
        )
        plt.gca().invert_yaxis()
        plt.grid(True)
        plt.show()

        return importance_df
