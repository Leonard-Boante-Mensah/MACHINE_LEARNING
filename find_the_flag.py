import codecademylib3_seaborn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

flags = pd.read_csv('flags.csv', header=0)
# print(flags.columns)
# print(flags.head(10))
# print('Andora is in Europe landmass')

labels = flags[['Landmass']]
# print(labels)
data = flags[["Red", "Green", "Blue", "Gold", "White", "Black", "Orange", "Circles", "Crosses","Saltires","Quarters","Sunstars", "Crescent","Triangle"]]
# print(data.head())
train_data, test_data, train_labels,  test_labels = train_test_split(data, labels,random_state = 1)

# print(train_data.shape)
# print(test_data.shape)
# print(train_labels.shape)
# print(test_labels.shape)
scores = []
for i in range(1, 21):
  tree = DecisionTreeClassifier(max_depth = i, random_state = 1)
  tree.fit(train_data, train_labels)
  scores.append(tree.score(test_data, test_labels))

plt.plot(range(1,21), scores)
plt.show()