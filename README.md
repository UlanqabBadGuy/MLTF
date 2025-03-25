# The Group Project - Machine Learning in Trading and Finance
## Inroduction
data file: data/Loan_Default/Loan_Default.csv

## Methodology
### Model selection
在“**贷款违约风险预测（Prediction of Loan Default Risk）**”这个任务中，我们面对的是一个**结构化表格数据上的二分类问题**（是否违约：0 or 1），目标是**精准预测高风险客户**。这类问题是金融风控中的经典场景。

---

#### ✅ **推荐的模型**

以下是业界、比赛和生产环境中都表现优秀、常用的模型：

| 模型名称        | 特点                                              | 是否推荐 |
|-----------------|---------------------------------------------------|----------|
| **LightGBM**     | 微软提出的 GBDT 优化版，速度快、性能强、适合大数据 | ⭐⭐⭐⭐⭐ 推荐 |
| **XGBoost**      | 最知名的 GBDT 实现，Kaggle 比赛常胜将军           | ⭐⭐⭐⭐     |
| **CatBoost**     | Yandex 提出的，天然支持类别变量                   | ⭐⭐⭐⭐     |
| **Random Forest**| 多个决策树的集合，稳定，抗过拟合                  | ⭐⭐⭐      |
| **Logistic Regression** | 基线模型，结果可解释性强（金融领域常用）    | ⭐⭐⭐      |
| **MLP（多层感知机）**   | 深度学习模型，需大量调参，适合复杂非线性     | ⭐⭐       |
| **SVM**          | 非常强的分类器，但在大数据上速度慢               | ⭐        |

---

#### 🧠 **模型选择建议**

1. **强烈建议作为主力模型：LightGBM**
   - 高效支持缺失值、类别变量（支持 `categorical_feature` 参数）
   - 自动特征选择
   - 训练快，预测也快
   - 可以处理不平衡数据

    ```python
    from lightgbm import LGBMClassifier
    
    model = LGBMClassifier(random_state=42)
    model.fit(X_train, y_train)
    ```

---

2. **XGBoost**
- 如果你希望对结果进行解释、调参细致，XGBoost 也很优秀

    ```python
    from xgboost import XGBClassifier
    
    model = XGBClassifier(random_state=42)
    model.fit(X_train, y_train)
    ```

---

3. **Logistic Regression**
- 若你对可解释性有强要求（比如银行风控需要“为什么客户被拒贷”） 可以输出系数、进行概率判断

    ```python
    from sklearn.linear_model import LogisticRegression
    
    model = LogisticRegression()
    model.fit(X_train, y_train)
    ```

---

#### ⚠️ 注意事项

数据不平衡问题（违约往往是少数）
- 如果你的标签中 `违约（Status=1）` 占比很少：
    - 可能要使用 `class_weight='balanced'` 参数
    - 或进行**上采样 / 下采样**
    - 或用 `SMOTE`（合成少数样本）

---

#### 📊 模型评估建议

仅仅看 `accuracy` 并不够，你还应该关注：

| 指标          | 含义                                     |
|---------------|------------------------------------------|
| **AUC-ROC**   | 衡量模型识别正负样本能力                 |
| **Precision** | 精确率，预测为违约中真正违约的比例       |
| **Recall**    | 召回率，所有违约中被模型识别的比例       |
| **F1-score**  | Precision 和 Recall 的调和平均           |
| **Confusion Matrix** | 真阳、假阳、真阴、假阴的具体数量  |

---

####  🧪 模型对比建议：

可以统一框架来跑多个模型对比（比如用 `sklearn` 或 `PyCaret`、`MLFlow`）

---

####  🎯 总结：建议使用路径

| 优先顺序 | 模型            | 适用场景                 |
|----------|------------------|--------------------------|
| 1        | LightGBM         | 默认首选，性能+速度兼顾 |
| 2        | XGBoost          | 高可控性、比赛常用       |
| 3        | CatBoost         | 类别变量多时很有优势     |
| 4        | LogisticRegression | 需要强解释性时使用     |

---

### Evaluation Metrics

### Results
