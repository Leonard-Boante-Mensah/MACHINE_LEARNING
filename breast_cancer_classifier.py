import codecademylib3_seaborn
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt

breast_cancer_data = load_breast_cancer()
X_train, validation_data, y_train, validation_test = train_test_split(breast_cancer_data.data, breast_cancer_data.target, test_size = 0.2, random_state = 1)

k_list = []
accuracies = []
for k in range(1, 100):
  classifier = KNeighborsClassifier(n_neighbors=k)
  classifier.fit(X_train, y_train)
  k_list.append(k)
  accuracies.append(classifier.score(validation_data, validation_test))

plt.plot(k_list, accuracies)
plt.xlabel('K VALUES')
plt.ylabel('Accuracy Scores')
plt.title('Breast Cancer Classifier Accuracy')
plt.show()

# print(X_train.shape)
# print(validation_data.shape)
# print(y_train.shape)
# print(validation_test.shape)

# print(breast_cancer_data.data[0])
# print(breast_cancer_data.feature_names)
# print(breast_cancer_data.target)
# print(breast_cancer_data.target_names)