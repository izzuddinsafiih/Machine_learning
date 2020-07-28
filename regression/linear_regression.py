import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv('../xlxs/Salary_Data.csv')

X = data.iloc[:, 0].values
y = data.iloc[:, 1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X = X.reshape(-1, 1)
X_train = X_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)

# 2.make model
model = LinearRegression()
model.fit(X_train, y_train)

y_fit = LinearRegression().fit(X, y).predict(X)
# prediction
y_pred = model.predict(X_test)

# 3.make score
accuracy = model.score(X_test, y_test)

# print(accuracy)

# # 4.make plot
plt.scatter(X, y)
plt.plot(X, y_fit, color='green')
plt.xlabel('Years os experience')
plt.ylabel('Salary')
plt.title('Salary based on experience')
plt.show()
# # plt.savefig('linear.png')
