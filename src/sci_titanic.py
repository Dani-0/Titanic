import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder
"""
Titanic ML Project 1: Predicting Survival Rate Given Age and Sex  (Logisitic Regression)
"""

# ------------ Load data and pre-process ------------#
print("Basic Examination of Data:\n")
data = pd.read_csv("./titanic_data/titanic.csv")
print(data.head())

print("Examine Null Data:\n")
print(data.isna().sum())  # see how many columns have NAs in them
print(f"Total rows: {len(data)}")

features = ["Age", "Sex"] # Maybe add "Pclass"??
label = "Survived"

data = data.dropna(axis=0, subset=features) # remove any features that have nan in them

X = data[features]
Y = data[label]

print("*"*50, "\nExamine Null Data in Features:\n")
print(X.isna().sum(),Y.isna().sum())

print("*"*50, "\nArray Shapes:\n")
print(f"Features array shape: {X.shape}")
print(f"Labels array shape: {Y.shape}")
print("*"*50)

class_1_samples = len(Y[Y==1])
class_0_samples = len(Y[Y==0])
total_samples = X.shape[0]
class_1_percent = (class_1_samples / total_samples) * 100   # 40
class_0_percent = (class_0_samples / total_samples) * 100   # 60

print(f"Class balances:\n\
\nPositive class: {class_1_samples} --> {class_1_percent:.3f}% \
\nNegative class: {class_0_samples} --> {class_0_percent:.3f}%")
print("*"*50)

numeric_columns = X.select_dtypes(include="number").columns
categoric_columns = X.select_dtypes(include="object").columns  # return string types

# ------------  Split into test-train sets ------------#
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, shuffle=True, random_state=42)


# ------------ Create feature transformers ------------#
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

numeric_transformer = Pipeline(
    steps =[
        ("numImputer", IterativeImputer(missing_values=np.nan)),
        ("numScaler", StandardScaler())
    ]
)

categoric_transformer = Pipeline(
    steps =[
        ("catImputer", IterativeImputer(missing_values=np.nan)),
        ("catOneHot", OneHotEncoder())
    ]
)



# ------------ Create ColumnTransformer  ------------#
from sklearn.compose import make_column_selector as selector
from sklearn.compose import ColumnTransformer

# If single categorical feature, OneHotEncoder, which requires
# a 1d array, must be specified explicitly in ColumnTransformer
if categoric_columns.shape[0] == 1:
    preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_transformer, numeric_columns),#selector(dtype_include="number")),
        ("categorical", OneHotEncoder(), categoric_columns)# selector(dtype_include="object"))
        ]
    )
else:
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_columns),#selector(dtype_include="number")),
            ("categorical", categoric_transformer(), categoric_columns)# selector(dtype_include="object"))
        ]
    )

# ------------ Create Models to Examine  ------------#
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

lr = LogisticRegression(penalty="l2", random_state=42)
rf = RandomForestClassifier(random_state=42, criterion="entropy")
knn = KNeighborsClassifier()
svc = SVC(random_state=42)

# ------------ Create Parameter Grids  ------------#

params_lr = {
    "classifier__C": [0.3, 0.5, 0.8, 1.0],
    "classifier__solver": ["lbfgs", "liblinear"],
    "classifier__max_iter": [100, 200, 500],
    "classifier": [lr]
}
params_rf = {
    "classifier__n_estimators": [10, 50, 100, 150],
    "classifier": [rf]
}
params_knn = {
    "classifier__n_neighbors": [5, 10, 15],
    "classifier__p": [1, 2],
    "classifier": [knn]
}
params_svc = {
    "classifier__C": [0.3, 0.5, 0.8, 1.0],
    "classifier__kernel": ["linear", "poly", "rbf", "sigmoid"],
    "classifier": [svc]
}

# params = [params_lr, params_rf, params_knn, params_svc]


# ----- Set up DataFrame to Hold GridSearchCV Results ----- #
index = []
scores = {"Accuracy": [], "Precision": [], "Recall": [], "F1": []}


# ------------ Perform GridSearchCV For Each Classifier ------------#
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
print("Commencing Grid Search:\n")

cv_strat = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# -------- Logistic Regression ------- # 
# -------- Create Full Pipeline ------- #
clf = Pipeline(
    steps = [("preprocessor", preprocessor),
             ("classifier", lr)]
)
grid_lf = GridSearchCV(estimator=clf,
                       cv=cv_strat,
                       param_grid=params_lr,
                       scoring=["accuracy", "precision", "recall", "f1"],
                       refit="accuracy",
                       verbose=1,
                       n_jobs=-1)

grid_lf.fit(x_train, y_train)

index += [grid_lf.best_estimator_["classifier"]]

scores["Accuracy"].append(grid_lf.cv_results_["mean_test_accuracy"][grid_lf.best_index_])
scores["Precision"].append(grid_lf.cv_results_["mean_test_precision"][grid_lf.best_index_])
scores["Recall"].append(grid_lf.cv_results_["mean_test_recall"][grid_lf.best_index_])
scores["F1"].append(grid_lf.cv_results_["mean_test_f1"][grid_lf.best_index_])


# -------- Random Forest ------- # 
# -------- Create Full Pipeline ------- #
clf = Pipeline(
    steps = [("preprocessor", preprocessor),
             ("classifier", rf)]
)

grid_rf = GridSearchCV(estimator=clf,
                       cv=cv_strat,
                       param_grid=params_rf,
                       scoring=["accuracy", "precision", "recall", "f1"],
                       refit="accuracy",
                       verbose=1,
                       n_jobs=-1)

grid_rf.fit(x_train, y_train)

index += [grid_rf.best_estimator_["classifier"]]

scores["Accuracy"].append(grid_rf.cv_results_["mean_test_accuracy"][grid_rf.best_index_])
scores["Precision"].append(grid_rf.cv_results_["mean_test_precision"][grid_rf.best_index_])
scores["Recall"].append(grid_rf.cv_results_["mean_test_recall"][grid_rf.best_index_])
scores["F1"].append(grid_rf.cv_results_["mean_test_f1"][grid_rf.best_index_])


# -------- K-Nearest Neighbours ------- # 
# -------- Create Full Pipeline ------- #
clf = Pipeline(
    steps = [("preprocessor", preprocessor),
             ("classifier", knn)]
)

grid_knn = GridSearchCV(estimator=clf,
                       cv=cv_strat,
                       param_grid=params_knn,
                       scoring=["accuracy", "precision", "recall", "f1"],
                       refit="accuracy",
                       verbose=1,
                       n_jobs=-1)

grid_knn.fit(x_train, y_train)

index += [grid_knn.best_estimator_["classifier"]]

scores["Accuracy"].append(grid_knn.cv_results_["mean_test_accuracy"][grid_knn.best_index_])
scores["Precision"].append(grid_knn.cv_results_["mean_test_precision"][grid_knn.best_index_])
scores["Recall"].append(grid_knn.cv_results_["mean_test_recall"][grid_knn.best_index_])
scores["F1"].append(grid_knn.cv_results_["mean_test_f1"][grid_knn.best_index_])


# -------- Support Vector Machine ------- # 
# -------- Create Full Pipeline ------- #
clf = Pipeline(
    steps = [("preprocessor", preprocessor),
             ("classifier", svc)]
)

grid_svc = GridSearchCV(estimator=clf,
                       cv=cv_strat,
                       param_grid=params_svc,
                       scoring=["accuracy", "precision", "recall", "f1"],
                       refit="accuracy",
                       verbose=1,
                       n_jobs=-1)

grid_svc.fit(x_train, y_train)

index += [grid_svc.best_estimator_["classifier"]]

scores["Accuracy"].append(grid_svc.cv_results_["mean_test_accuracy"][grid_svc.best_index_])
scores["Precision"].append(grid_svc.cv_results_["mean_test_precision"][grid_svc.best_index_])
scores["Recall"].append(grid_svc.cv_results_["mean_test_recall"][grid_svc.best_index_])
scores["F1"].append(grid_svc.cv_results_["mean_test_f1"][grid_svc.best_index_])

# ------------ Display GridSearch Results (best of each classifier) ------------#
print("*"*50, "\nGridSearch Results:")
df_scores = pd.DataFrame(scores, index=index)
df_scores.to_csv("./results.csv")
print(df_scores)


# ------------ Create Best Parameter Models + Fit & Predict ------------#
from sklearn.metrics import accuracy_score

print("*"*50, "\nFinal Accuracy Scores:\n")
# -------- Logistic Regression ------- # 
lf_best = LogisticRegression(C=0.3, random_state=42)
clf = Pipeline(
    steps = [("preprocessor", preprocessor),
             ("classifier", lf_best)]
)
clf.fit(x_train, y_train)
pred_lf = clf.predict(x_test)
acc_lf = accuracy_score(y_test, pred_lf)* 100
print(f"Logistic Regression Accuracy: {acc_lf:.4f}%")

# -------- Random Forest ------- # 
rf_best = RandomForestClassifier(criterion='entropy', n_estimators=150, random_state=42)
clf = Pipeline(
    steps = [("preprocessor", preprocessor),
             ("classifier", rf_best)]
)
clf.fit(x_train, y_train)
pred_rf = clf.predict(x_test)
acc_rf = accuracy_score(y_test, pred_rf)* 100
print(f"Random Forest Accuracy: {acc_rf:.4f}%")

# -------- K-Nearest Neighbours ------- # 
knn_best = KNeighborsClassifier(p=1)
clf = Pipeline(
    steps = [("preprocessor", preprocessor),
             ("classifier", knn_best)]
)
clf.fit(x_train, y_train)
pred_knn = clf.predict(x_test)
acc_knn = accuracy_score(y_test, pred_knn)* 100
print(f"K-Nearest Neighbours  Accuracy: {acc_knn:.4f}%")

# -------- Support Vector Machine ------- # 
svc_best = SVC(C=0.3, kernel='linear', random_state=42)
clf = Pipeline(
    steps = [("preprocessor", preprocessor),
             ("classifier", svc_best)]
)
clf.fit(x_train, y_train)
pred_svc = clf.predict(x_test)
acc_svc = accuracy_score(y_test, pred_svc)* 100
print(f"Support Vector Machine Accuracy: {acc_svc:.4f}%")



# TODO Feature importance/feature engineering?
# TODO Get weights for linear models

# ------------ Create  ------------#