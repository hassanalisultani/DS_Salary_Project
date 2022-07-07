import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

import statsmodels.api as sm
import pickle

df = pd.read_csv('eda_data.csv')


# chose relevent columns
# print(df.columns)

df_model = df[['avg_salary','Rating','Size','Type of ownership','Industry','Sector','Revenue','num_comp','hourly','employer_provided',
'job_state','same_state','age','python_yn','spark','aws','excel','job_simp','seniority','desc_len']]


# get dummy data
df_dum = pd.get_dummies(df_model)


# train test split
X = df_dum.drop('avg_salary', axis=1)
y = df_dum.avg_salary.values

X_test, X_train, y_test, y_train = train_test_split(X, y, test_size=0.2, random_state=42)


# muultiple linear regression
X_sm = X = sm.add_constant(X)
model = sm.OLS(y, X_sm)
model.fit().summary()

lm = LinearRegression()
lm.fit(X_train, y_train)
np.mean(cross_val_score(lm, X_train, y_train, scoring='neg_mean_absolute_error', cv=3))


# lasso regression
lm_l = Lasso(alpha=0.3)
lm_l.fit(X_train, y_train)
np.mean(cross_val_score(lm_l, X_train, y_train, scoring='neg_mean_absolute_error', cv=3))

alpha = []
error = []

for i in range(1,100):
    alpha.append(i/100)
    lml = Lasso(alpha=(i/100))
    error.append(np.mean(cross_val_score(lml, X_train, y_train, scoring='neg_mean_absolute_error', cv=3)))

plt.plot(alpha, error)

err = tuple(zip(alpha, error))
df_err = pd.DataFrame(err, columns=['alpha', 'eroor'])
df_err[df_err == max(df_err)]


# random forest
rf = RandomForestRegressor()
np.mean(cross_val_score(rf, X_train, y_train, scoring='neg_mean_absolute_error', cv=3))


# tune models GridsearchCV
# parameters = {'n_estimators':range(10,300,10), 'criterion':('mse','mae'), 'max_features':('auto','sqrt','log2')}
parameters = {'n_estimators':range(10,300,10), 'criterion':('squared_error','absolute_error'), 'max_features':('auto','sqrt','log2')}

gs = GridSearchCV(rf, parameters, scoring='neg_mean_absolute_error', cv=3)
gs.fit(X_train, y_train)

gs.best_score_
gs.best_estimator_


# test ensembles
tpred_lm = lm.predict(X_test)
tpred_lml = lm_l.predict(X_test)
tpred_rf = gs.best_estimator_.predict(X_test)


print(mean_absolute_error(y_test, tpred_lm))
print(mean_absolute_error(y_test, tpred_lml))
print(mean_absolute_error(y_test, tpred_rf))

print(mean_absolute_error(y_test, (tpred_lm+tpred_rf)/2))


# # pickle
# pickl = {'model':gs.best_estimator_}
# pickle.dump(pickl, open('model_file' + ".p", "wb"))

# file_name = "model_file.p"
# with open(file_name, 'rb') as pickled:
#     data = pickle.load(pickled)
#     model = data['model']

# model.predict(np.array(list(X_test.iloc[1,:])).reshape(1,-1))[0]

# list(X_test.iloc[1,:])