import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

data = pd.read_csv('../xlxs/datasets_7068_10152_advertising.csv')

X = data.iloc[:, [0, 1, 2, 3, 6, 7]].values
y = data.iloc[:, 9].values

# ENCODE STRING CODER
ohe = OneHotEncoder(drop='first', sparse=False)
X_country = ohe.fit_transform(X[:, 5].reshape(-1, 1))
# scale numeric data
sc_X = StandardScaler()
X_numeric = sc_X.fit_transform(X[:, [0, 1, 2, 3, 4]])

X = np.hstack((X_numeric, X_country))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression()
# model = SVC(kernel='rbf')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(model.score(X_test, y_test))

cm = confusion_matrix(y_test, y_pred)
print(cm)