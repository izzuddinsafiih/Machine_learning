import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

data = pd.read_csv('../xlxs/Social_Network_Ads.csv')

X = data.iloc[:, 1:4].values
y = data.iloc[:, 4].values

# encode string gender
ohe = OneHotEncoder(drop='first', sparse=False)
X_gender = ohe.fit_transform(X[:, 1].reshape(-1, 1))

# scale numeric data
sc_X = StandardScaler()
X_numeric = sc_X.fit_transform(X[:, 1:3])

X = np.hstack((X_numeric, X_gender))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# model = LogisticRegression()
model = SVC(kernel='rbf')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(model.score(X_test, y_test))

cm = confusion_matrix(y_test, y_pred)
print(cm)