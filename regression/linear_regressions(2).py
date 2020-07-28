import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


data = pd.read_csv('../xlxs/datasets_1256_2242_train.csv')

X = data.iloc[:, 0].values
y = data.iloc[:, 1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X = X.reshape(-1, 1)
X_train = X_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)

model = LinearRegression()
model.fit(X_train, y_train)

y_fit = LinearRegression().fit(X, y).predict(X)
# prediction
# y_pred = model.predict(X_test)


accuracy = model.score(X_test, y_test)
#
# print(accuracy)
plt.scatter(X, y)
plt.plot(X, y_fit, color='blue')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Dataset')
plt.show()
