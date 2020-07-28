import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

data = pd.read_csv('../xlxs/Position_Salaries.csv')

# 1D = [1, 2, 3, 4]
# 2D = [[1, 2], [2, 3], [3, 4]]

X = data.iloc[:, 1].values.reshape(-1, 1)
y = data.iloc[:, 2].values

poly_reg_model = PolynomialFeatures(degree=3)
X_poly = poly_reg_model.fit_transform(X)
# poly_reg_model.fit(X_poly)
lin_reg_model = LinearRegression()
lin_reg_model.fit(X_poly, y)

y_fit_poly = lin_reg_model.predict(poly_reg_model.transform(X))

# testing
test_level = np.array([5.5, 6.5]).reshape(-1, 1)
y_pred = lin_reg_model.predict(poly_reg_model.transform(test_level))

print()

# plotting
plt.scatter(X, y)
plt.plot(X, y_fit_poly, color='black')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.title('Salary based on Level')
plt.show()


