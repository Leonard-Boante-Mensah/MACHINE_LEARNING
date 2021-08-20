def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

income_data = pd.read_csv('income.csv', header = 0, delimiter = ', ')

income_data['sex-int'] = income_data['sex'].apply(lambda row: 0 if row == 'Male' else 1)
#print(income_data['native-country'].value_counts())
income_data['country-int'] = income_data['native-country'].apply(lambda row: 0 if row =='United-States' else 1) 
print(income_data['country-int'].value_counts())

# print(income_data.iloc[0])
labels = income_data[['income']]
data = income_data[['age', 'capital-gain','capital-loss', 'hours-per-week', 'sex-int', 'country-int']]

train_data, test_data, train_labels, test_labels = train_test_split(data, labels, random_state=1)
forest = RandomForestClassifier(random_state = 1)
forest.fit(train_data, train_labels)
print(forest.feature_importances_)
print(forest.score(test_data, test_labels))



# print(train_data.shape)
# print(test_data.shape)
# print(train_labels.shape)
# print(test_labels.shape)
#print(income_data.columns)
# print(data.head())