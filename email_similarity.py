from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

emails = fetch_20newsgroups()
print(emails.target_names)

categories = ['rec.sport.baseball', 'rec.sport.hockey']
emails = fetch_20newsgroups(categories=categories)

# print(emails.data[5])
# print(emails.target[5])
# print(emails.target_names[1])

train_emails = fetch_20newsgroups(categories=['comp.sys.ibm.pc.hardware','rec.sport.hockey'], subset='train', shuffle=True, random_state = 108)
test_emails = fetch_20newsgroups(categories=['comp.sys.ibm.pc.hardware','rec.sport.hockey'], subset='test', shuffle=True, random_state = 45)

counter = CountVectorizer()
counter.fit(train_emails.data + test_emails.data)

train_counts = counter.transform(train_emails.data)
test_counts = counter.transform(test_emails.data)

classifier = MultinomialNB()
classifier.fit(train_counts, train_emails.target)


print(classifier.score(test_counts, test_emails.target))


















