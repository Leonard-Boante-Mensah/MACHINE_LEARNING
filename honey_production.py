import codecademylib3_seaborn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.linear_model import LinearRegression

df = pd.read_csv("https://content.codecademy.com/programs/data-science-path/linear_regression/honeyproduction.csv")

prod_per_year = df.groupby('year').totalprod.mean().reset_index()

X = prod_per_year.year
y = prod_per_year.totalprod

X = X.values.reshape(-1,1)

my_model = LinearRegression()
my_model.fit(X, y)

print(my_model.coef_)
print(my_model.intercept_)

y_predict = my_model.predict(X)

X_future = np.array(range(2013, 2051))

print(X_future)
X_future = X_future.reshape(-1,1)
print(X_future)

future_predict = my_model.predict(X_future)

# print(df.head())
# print(df.info())
plt.plot(X_future, future_predict)
plt.show()

plt.clf()

plt.scatter(X, y)
plt.plot(X, y_predict)
plt.show()



