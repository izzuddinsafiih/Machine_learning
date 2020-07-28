import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# 1.data preprocessing
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

dataset = pd.read_csv('../xlxs/Social_Network_Ads.csv')

X = dataset.iloc[:, 1:4].values
y = dataset.iloc[:, 4].values
# avoid dummy variable trap
ohe = OneHotEncoder(sparse=False, drop='first')
# 3 = 0,1,2
X_gender = ohe.fit_transform(X[:, [0]])
# standard scaller
sc_X = StandardScaler()
X_numeric = sc_X.fit_transform(X[:, 1:3])
X = np.hstack((X_numeric, X_gender))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# 2. make model
model = SVC(kernel='rbf')
model.fit(X_train, y_train)
# 4.make visual
visual_x_y = dataset.iloc[:, [2, 3]].values
genders = dataset.iloc[:, 1].values
y_pred = model.predict(X_test)

# make plot
legend_elements = [
    Line2D([0], [0], c='w', marker='s', label='Male', markerfacecolor='black', markersize=10),
    Line2D([0], [0], c='w', marker='^', label='Female', markerfacecolor='black', markersize=10),
    Patch(facecolor='red', edgecolor='black', label='Not Purchased'),
    Patch(facecolor='orange', edgecolor='black', label='Purchased')
]

fig, ax = plt.subplots()
fig.set_size_inches((10, 7))
plt.scatter(visual_x_y[(y_pred == 0) & (genders == 'Male'), 0],
            visual_x_y[(y_pred == 0) & (genders == 'Male'), 1],
            marker='s',
            c='orange',
            label='Not purchased male')
plt.scatter(visual_x_y[(y_pred == 0) & (genders == 'Female'), 0],
            visual_x_y[(y_pred == 0) & (genders == 'Female'), 1],
            marker='^',
            c='orange',
            s=70,
            label='Not purchased female')
plt.scatter(visual_x_y[(y_pred == 1) & (genders == 'Male'), 0],
            visual_x_y[(y_pred == 1) & (genders == 'Male'), 1],
            marker='s',
            c='red',
            label='Not purchased male')
plt.scatter(visual_x_y[(y_pred == 1) & (genders == 'Female'), 0],
            visual_x_y[(y_pred == 1) & (genders == 'Female'), 1],
            marker='^',
            c='orange',
            s=70,
            label='Not purchased female')
plt.title('Prediction purchase or not')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
ax.legend(handles=legend_elements)

plt.show()
