import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.inspection import permutation_importance

"""
Titanic ML Project 1: Predicting Survival Rate Given Age and Sex  (Logisitic Regression)
"""

# ------------ Define plotting function for permutation importance ------------#
def plot_permutation_importance(clf, X, y, ax):
    result = permutation_importance(clf, X, y, n_repeats=10, random_state=42, n_jobs=2)
    perm_sorted_idx = result.importances_mean.argsort()

    ax.boxplot(
        result.importances[perm_sorted_idx].T,
        vert=False,
        labels=X.columns[perm_sorted_idx],
    )
    ax.axvline(x=0, color="k", linestyle="--")
    return ax


# ------------ Load data and pre-process ------------#
print("Basic Examination of Data:\n")
DATADIR = "./data/titanic.csv"
data = pd.read_csv(DATADIR)
print(data.head())

print("Examine Null Data:\n")
print(data.isna().sum())  # see how many columns have NAs in them
print(f"Total rows: {len(data)}")

# features = ["Age", "Sex", "Pclass", "SibSp", "Parch"] 
# label = "Survived"
# X = data[features]
# Y = data[label]

# remove cols that have don't have majority non nan
data = data.dropna(axis=1, thresh=0.8*len(data)) 

# remove any rows that have nan in them; excl age column
subset_cols = data.columns[data.columns != "Age"]
data = data.dropna(axis=0, subset=subset_cols) 

# Remove unnecessary categorical columns
data = data.drop(columns=["Name", "Ticket"])

X = pd.DataFrame(data.values[:,2:], columns=data.columns[2:])
Y = pd.Series(data.values[:,1], name=data.columns[1])

print("*"*50, "\nExamine Null Data in Features:\n")
print(X.isna().sum(),Y.isna().sum())

print("*"*50, "\nArray Shapes:\n")
print(f"Features array shape: {X.shape}")
print(f"Labels array shape: {Y.shape}")
print("*"*50)

class_1_samples = len(Y[Y==1])
class_0_samples = len(Y[Y==0])
total_samples = X.shape[0]
class_1_percent = (class_1_samples / total_samples) * 100   # ~40
class_0_percent = (class_0_samples / total_samples) * 100   # ~60

print(f"Class balances:\n\
\nPositive class: {class_1_samples} --> {class_1_percent:.3f}% \
\nNegative class: {class_0_samples} --> {class_0_percent:.3f}%")
print("*"*50)

numeric_columns = X.infer_objects().select_dtypes(include="number").columns
categoric_columns = X.infer_objects().select_dtypes(include="object").columns  # return string types

# ------------  Split into test-train sets ------------#

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, shuffle=True, random_state=42)

# ----------- Label Encode output labels ------------#

lb = LabelEncoder()
lb.fit(y_train)
y_train = lb.transform(y_train)
y_test = lb.transform(y_test)

# ------------ Create feature transformers ------------#

numeric_transformer = Pipeline(
    steps =[
        ("numImputer", SimpleImputer(missing_values=np.nan)),
        ("numScaler", StandardScaler())
    ]
)

categoric_transformer = Pipeline(
    steps =[
        ("catOneHot", OneHotEncoder())
    ]
)

feature_transformer = Pipeline(
    steps =[
        ("FeatureSelect", SelectKBest(score_func=f_classif, k=6))
    ]
)

# ------------ Create ColumnTransformer  ------------#

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
            ("categorical", categoric_transformer, categoric_columns) # selector(dtype_include="object"))
        ]
    )

# ------------ Create Models to Examine  ------------#

lr = LogisticRegression(penalty="l2", random_state=42)
rf = RandomForestClassifier(random_state=42, criterion="entropy")
knn = KNeighborsClassifier()
svc = SVC(random_state=42)

classifiers = [lr, rf, knn, svc]

# ------------ Create Parameter Grids  ------------#

params_lr = {
    "classifier__C": [0.3, 0.5, 0.8, 1.0],
    "classifier__solver": ["lbfgs", "liblinear"],
    "classifier__max_iter": [100, 200, 500]
    # "classifier": [lr]
}
params_rf = {
    "classifier__n_estimators": [10, 50, 100, 150],
    "classifier__min_samples_leaf": [5, 10, 15, 20] # added to reduce overfitting
    # "classifier": [rf]
}
params_knn = {
    "classifier__n_neighbors": [5, 10, 15],
    "classifier__p": [1, 2]
    # "classifier": [knn]
}
params_svc = {
    "classifier__C": [0.3, 0.5, 0.8, 1.0],
    "classifier__kernel": ["linear", "poly", "rbf", "sigmoid"]
    # "classifier": [svc]
}

params = [params_lr, params_rf, params_knn, params_svc]

# ----- Set up DataFrame to Hold GridSearchCV Results ----- #

index = []
scores = {"Accuracy": [], "Precision": [], "Recall": [], "F1": []}

# ------------ Perform GridSearchCV For Each Classifier ------------#

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
print("Commencing Grid Search:\n")

cv_strat = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)

for clf, param in zip(classifiers, params):

    pipe = Pipeline(
        steps = [("preprocessor", preprocessor),
                ("featureSelector", feature_transformer),
                ("classifier", clf)] 
    )
    grid = GridSearchCV(estimator=pipe,
                        cv=cv_strat,
                        param_grid=param,
                        scoring=["accuracy", "precision", "recall", "f1"],
                        refit="accuracy",
                        error_score="raise",
                        verbose=3,
                        n_jobs=-1)

    grid.fit(x_train, y_train)

    index += [grid.best_estimator_["classifier"]]

    scores["Accuracy"].append(grid.cv_results_["mean_test_accuracy"][grid.best_index_])
    scores["Precision"].append(grid.cv_results_["mean_test_precision"][grid.best_index_])
    scores["Recall"].append(grid.cv_results_["mean_test_recall"][grid.best_index_])
    scores["F1"].append(grid.cv_results_["mean_test_f1"][grid.best_index_])

# ------------ Display GridSearch Results (best of each classifier) ------------#
    
print("*"*50, "\nGridSearch Results:")
df_scores = pd.DataFrame(scores, index=index)
df_scores.to_csv("./gridsearch_results.csv")
print(df_scores)

# ------------ Create Best Parameter Models ------------#

lr_best = LogisticRegression(C=0.3, random_state=42, solver="liblinear")
rf_best = RandomForestClassifier(criterion='entropy', min_samples_leaf=10, n_estimators=10, random_state=42)
knn_best = KNeighborsClassifier()
svc_best = SVC(C=0.5, random_state=42)

best_models = [lr_best, rf_best, knn_best, svc_best]

# ------------ Fit and Predict using best models  ------------#
print("*"*50, "\nFinal Accuracy Scores:\n")

index = []
scores = {"Accuracy": [], "Precision": [], "Recall": [], "F1": []}


for model in best_models:
    pipe = Pipeline(
    steps = [("preprocessor", preprocessor),
             ("featureSelector", feature_transformer),
             ("classifier", model)]
    )
    pipe.fit(x_train, y_train)
    pred = pipe.predict(x_test)
    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred)
    rec = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)

    index += [str(model)]
    scores["Accuracy"].append(acc)
    scores["Precision"].append(prec)
    scores["Recall"].append(rec)
    scores["F1"].append(f1)

    # ------------ Plot permutation importances for supervised models  ------------#
    if model != knn_best:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,8))
        ax1.set_xlabel("Decrease in accuracy score <<Training Set>>")
        ax2.set_xlabel("Decrease in accuracy score <<Test Set>>")
        plot_permutation_importance(pipe, x_train, y_train, ax1)
        plot_permutation_importance(pipe, x_test, y_test, ax2)
        _ = fig.tight_layout
        plt.show()


# ------------ Display Test Results  ------------#

df_test_scores = pd.DataFrame(scores, index=index)
df_test_scores.to_csv("./test_results.csv")
print(df_test_scores)
