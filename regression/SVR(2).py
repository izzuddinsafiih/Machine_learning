import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

data = pd.read_csv('xlxs/forestfires.csv')

X = data.iloc[:, 2:8].values
y = data.iloc[:, 8].values.reshape(-1, 1)

sc_X = StandardScaler()
sc_y = StandardScaler()

X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

# model
svr_model = SVR(kernel='poly')
svr_model.fit(X, y)

y_fit = svr_model.predict(X)
# plot
plt.scatter(X, y)
plt.plot(X, y_fit, color='blue')
plt.show()
