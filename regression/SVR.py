# Support Vector Regressions

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# data preprocessing
data = pd.read_csv('../xlxs/Position_Salaries.csv')

# 1D = [1, 2, 3, 4]
# 2D = [[1, 2], [2, 3], [3, 4]]

X = data.iloc[:, 1].values.reshape(-1, 1)
y = data.iloc[:, 2].values

sc_X = StandardScaler()
sc_y = StandardScaler()

X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y.reshape(-1, 1))

# model
# 3 types: 'poly','linear' & 'rbf'
svr_model = SVR(kernel='linear')
svr_model.fit(X, y)

# Standard Scaller
y_fit = svr_model.predict(X)
# plot
plt.scatter(X, y)
plt.plot(X, y_fit, color='blue')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.title('Salary based on Level')
plt.show()

