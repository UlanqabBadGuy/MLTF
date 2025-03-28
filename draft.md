## COMP7409A Machine Learning in Trading and Finance

### 项目主题：基于机器学习的贷款违约风险预测

### 1. 项目背景
贷款违约风险预测是金融领域的经典问题，其目标是提前识别高风险客户，帮助金融机构制定有效的风险管理策略。本项目旨在利用机器学习方法，基于历史贷款数据进行精准预测，以优化贷款审批流程，降低违约带来的损失。

### 2. 数据分析与EDA
本项目使用来自Kaggle的Loan Default数据集，包含贷款人财务状况、信用评分、贷款金额及违约状态（违约=1，未违约=0）等信息。

- 数据集存在信息泄露问题，尤其是`Interest_rate_spread`、`rate_of_interest`、`Upfront_charges` 等特征的缺失值与违约状态直接相关，故在建模过程中删除这些特征。
- 数据呈现类别不平衡，违约样本占比较低。
- EDA阶段进行了缺失值可视化、数据分布分析、违约率分析以及变量之间相关性分析。

### 3. 特征工程
为提升模型性能，执行了以下特征工程步骤：
- 删除信息泄露特征 (`Interest_rate_spread`, `rate_of_interest`, `Upfront_charges`)
- 缺失值填充（使用中位数或众数填补）
- 类别变量编码（独热编码）
- 数据标准化处理

### 4. 模型选择与训练
本项目重点测试了五种常见模型：LightGBM、XGBoost、CatBoost、Logistic Regression 和 MLP神经网络。根据业界经验和Kaggle比赛结果，LightGBM和XGBoost通常表现突出。

### 5. 模型优化与参数调优
通过GridSearchCV进行超参数优化，各模型最优参数为：
- LightGBM：`{'num_leaves': 128, 'learning_rate': 0.05, 'max_depth': -1}`
- XGBoost：`{'learning_rate': 0.05, 'max_depth': 10, 'n_estimators': 200}`
- CatBoost：`{'learning_rate': 0.1, 'depth': 8}`
- MLP：`{'batch_size': 64, 'lr': 0.001, 'epochs': 15}`
- Logistic Regression：`{'C': 10.0}`

### 6. 模型评估与结果分析
采用以下评估指标进行全面分析：
- AUC-ROC
- Precision（精确率）
- Recall（召回率）
- F1 Score
- Confusion Matrix（混淆矩阵）

评估结果表明LightGBM表现最优（AUC=0.8981），其次为XGBoost（AUC=0.8968）。Logistic Regression表现一般（AUC=0.8400）。

### 7. 特征重要性分析
特征重要性分析显示以下变量对预测违约风险具有显著作用：
- LTV（贷款价值比）
- Income（收入水平）
- Credit Score（信用评分）
- Loan Amount（贷款金额）
- Property Value（房产价值）

### 8. 可视化展示
使用Plotly和Matplotlib等库，生成了模型性能比较图、AUC对比图、超参数调优热图、特征重要性图，以及准确率与AUC的交互图。

### 9. 项目结论
本项目成功应用机器学习模型预测贷款违约风险，LightGBM模型表现最佳，可为实际金融决策提供有力支持。建议金融机构采用该模型并结合重要特征进行风险管控。

### 10. 未来改进方向
- 引入更多维度数据，如客户行为数据或外部经济指标
- 考虑更先进的特征工程技术，如自动特征生成
- 使用集成模型进一步提升性能

