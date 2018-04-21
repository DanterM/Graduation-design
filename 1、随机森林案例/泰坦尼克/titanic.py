import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


# 数据导入以及数据清洗
train = pd.read_csv("/Users/Jarvis/PycharmProjects/titanic/train.csv", dtype={"Age": np.float64},)

train.head(10)
def harmonize_data(titanic):
    # 填充空数据 和 把string数据转成integer表示
    # 对于年龄字段发生缺失，我们用所有年龄的替代
    titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())

    titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
    # 性别男： 用0替代
    titanic.loc[titanic["Sex"] == "female", "Sex"] = 1
    # 性别女： 用1替代
    titanic["Embarked"] = titanic["Embarked"].fillna("S")

    titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
    titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
    titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2

    titanic["Fare"] = titanic["Fare"].fillna(titanic["Fare"].median())

    return titanic

train_data = harmonize_data(train)





# len(train_data)
# 列出对生存结果有影响的字段
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

# 存放不同参数取值，以及对应的精度，每一个元素都是一个三元组(a, b, c)
results = []

# 最小叶子结点的参数取值
sample_leaf_options = list(range(1, 500, 3))

# 决策树个数参数取值
n_estimators_options = list(range(1, 1000, 5))
groud_truth = train_data['Survived'][601:]

# n_estimators：决策树数量
# min_samples_leaf:最小样本叶片大小
for leaf_size in sample_leaf_options:
    for n_estimators_size in n_estimators_options:
        alg = RandomForestClassifier(min_samples_leaf=leaf_size, n_estimators=n_estimators_size, random_state=50)
        alg.fit(train_data[predictors][:600], train_data['Survived'][:600])
        predict = alg.predict(train_data[predictors][601:])
        # 用一个三元组，分别记录当前的 min_samples_leaf，n_estimators， 和在测试数据集上的精度
        results.append((leaf_size, n_estimators_size, (groud_truth == predict).mean()))
        # 真实结果和预测结果进行比较，计算准确率
        print((groud_truth == predict).mean())

# 打印精度最大的那一个三元组
print(max(results, key=lambda x: x[2]))

#RF特征
#底层逻辑代码
#可视化