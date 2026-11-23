# 코드_공유.py
# 작업자 A, B, Admin 이 함께 사용하는 파일입니다.

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV

# 2013년 뉴욕 출발 항공편 표본 50,000건입니다.
# 항공편 정보에 기상 정보와 기체 정보를 미리 결합해 두었습니다.
df = pd.read_csv('flights_ml.csv')

NUM = ['month', 'day', 'sched_hour', 'distance', 'temp', 'humid',
       'wind_speed', 'precip', 'visib', 'seats', 'engines']
CAT = ['carrier', 'origin']

X = pd.get_dummies(df[NUM + CAT], columns=CAT)
X = X.fillna(X.median(numeric_only=True))
y = df['is_delayed']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(X_train.shape, X_test.shape)


####### A 작업자 작업 수행 #######

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, accuracy_score

dt_params = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 7],
    'min_samples_leaf': [1, 5, 20],
}

dt_grid = GridSearchCV(DecisionTreeClassifier(random_state=42),
                       param_grid=dt_params, cv=5, scoring='f1', n_jobs=-1)
dt_grid.fit(X_train, y_train)

print('[DecisionTree] 최적 하이퍼파라미터 :', dt_grid.best_params_)
print('[DecisionTree] CV F1 (best)      :', round(dt_grid.best_score_, 4))

dt_best = dt_grid.best_estimator_
dt_pred = dt_best.predict(X_test)
dt_f1 = f1_score(y_test, dt_pred)
dt_acc = accuracy_score(y_test, dt_pred)
print('[DecisionTree] Test F1       :', round(dt_f1, 4))
print('[DecisionTree] Test Accuracy :', round(dt_acc, 4))

# Feature Importance 상위 10개 시각화
dt_imp = pd.Series(dt_best.feature_importances_, index=X_train.columns)
dt_top10 = dt_imp.sort_values(ascending=False).head(10)

plt.figure(figsize=(8, 5))
dt_top10.sort_values().plot(kind='barh')
plt.title('DecisionTree Feature Importance (Top 10)')
plt.xlabel('importance')
plt.tight_layout()
plt.show()


####### B 작업자 작업 수행 #######

from xgboost import XGBClassifier

xgb_params = {
    'max_depth': [3, 6, 9],
    'n_estimators': [100, 300],
    'learning_rate': [0.1, 0.05],
}

xgb_grid = GridSearchCV(XGBClassifier(random_state=42, eval_metric='logloss'),
                        param_grid=xgb_params, cv=5, scoring='f1', n_jobs=-1)
xgb_grid.fit(X_train.astype(float), y_train)

print('[XGBoost] 최적 하이퍼파라미터 :', xgb_grid.best_params_)
print('[XGBoost] CV F1 (best)      :', round(xgb_grid.best_score_, 4))

xgb_best = xgb_grid.best_estimator_
xgb_pred = xgb_best.predict(X_test.astype(float))
xgb_f1 = f1_score(y_test, xgb_pred)
xgb_acc = accuracy_score(y_test, xgb_pred)
print('[XGBoost] Test F1       :', round(xgb_f1, 4))
print('[XGBoost] Test Accuracy :', round(xgb_acc, 4))

# Feature Importance 상위 10개 시각화
xgb_imp = pd.Series(xgb_best.feature_importances_, index=X_train.columns)
xgb_top10 = xgb_imp.sort_values(ascending=False).head(10)

plt.figure(figsize=(8, 5))
xgb_top10.sort_values().plot(kind='barh')
plt.title('XGBoost Feature Importance (Top 10)')
plt.xlabel('importance')
plt.tight_layout()
plt.show()
