import codecademylib3_seaborn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the passenger data
passengers = pd.read_csv('passengers.csv')
print(passengers.head())


# Update sex column to numerical
sex_column = {'male': 0, 'female': 1}
passengers['Sex'] = passengers.Sex.replace(sex_column)
print(passengers.head())

# Fill the nan values in the age column

passengers.Age.fillna(np.mean(passengers.Age), inplace=True)
print(passengers['Age'].unique)
# Create a first class column
passengers['FirstClass'] = passengers['Pclass'].apply(lambda x: 1 if x == 1 else 0)

# Create a second class column
passengers['SecondClass'] = passengers['Pclass'].apply(lambda x: 1 if x == 2 else 0)
print(passengers.head(10))


# Select the desired features
features = passengers[['Sex', 'Age', 'FirstClass', 'SecondClass']]
survival = passengers['Survived']


# Perform train, test, split
x_train, x_test, y_train,  y_test = train_test_split(features, survival)


# Scale the feature data so it has mean = 0 and standard deviation = 1
scale = StandardScaler()
train_features = scale.fit_transform(x_train)
test_features = scale.transform(x_test)


# Create and train the model
model = LogisticRegression()
model.fit(x_train, y_train)


# Score the model on the train data
train_score = model.score(x_train, y_train)
print(train_score)


# Score the model on the test data
test_score = model.score(x_test, y_test)
print(test_score)

print(model.coef_)
# Analyze the coefficients


# Sample passenger features
Jack = np.array([0.0,20.0,0.0,0.0])
Rose = np.array([1.0,17.0,1.0,0.0])
You = np.array([1.0,18.0,0.0,1.0])

# Combine passenger arrays
sample_passengers = np.array([Jack, Rose, You])

# Scale the sample passenger features
sample_passengers = scale.transform(sample_passengers)
print(sample_passengers)

# Make survival predictions!
print(model.predict_proba(sample_passengers))





