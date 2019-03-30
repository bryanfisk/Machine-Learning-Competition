print("Importing modules.")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import re

def ezip(ls1, ls2):
    count = 0
    e = [] 
    for index in range(len(ls1)):
        e.append([count, ls1[index], ls2[index]])
        count += 1
    return e

def readfile(file):
    input = []
    with open(file, encoding = "utf-8 ") as f:
        for line in f:
                input.append(line.strip())
    return input

print("Reading files.")
train_input = readfile("train.txt")
test_input = tuple(readfile("test.txt"))
words = readfile("words.txt")
words = tuple([tuple(k.split(',')) for k in words])

pattern = re.compile(' ?\{([^\|]+)\|([^\}]+)\} ?')
print("Formatting test set.")

X_tests = []
for couple in words:
    X = []
    for index, phrase in enumerate(test_input):
        m = re.search(pattern, phrase)
        if couple[0] in [m.group(1), m.group(2)] or couple[1] in [m.group(1), m.group(2)]:
            X.append(phrase.replace(m.group(0), ""))
    X_tests.append(X)

def format_trains(i):
    print("Formatting train set.")
    X_trains = []
    y_trains = []
    for couple in words:
        X = []
        y = []
        for phrase in i:
            if couple[0] in phrase.split(' '):
                X.append(phrase)
                y.append(couple[0])
            elif couple[1] in phrase.split(' '):
                X.append(phrase)
                y.append(couple[1])
        X_trains.append(X)
        y_trains.append(y)
    return X_trains, y_trains

results = []
#clf = SGDClassifier(loss = 'log', penalty = 'l2', alpha = 1e-3, random_state = 42, max_iter = 15, tol = 1e-3)
clf = [SGDClassifier(), ]
for index, X_train_set, X_test_set in ezip(X_trains, X_tests):
    vectorizer = TfidfVectorizer(analyzer = 'word', stop_words = list(words[index]), ngram_range = (2,2))
    X_train_tfidf = vectorizer.fit_transform(X_train_set, y_trains[index])
    X_test_tfidf = vectorizer.transform(X_test_set)

for index, X_train_tfidf, X_test_tfidf in ezip(X_trains, X_tests):
    clf.fit(X_train_tfidf, y_trains[index])
    results.append(clf.predict_proba(X_test_tfidf))
    print(clf.classes_)
    print(clf.predict_proba(X_test_tfidf)) 

count = 1
with open("results.csv", "w", encoding  = "utf-8") as file:
    for table in results:
        for value in table[:, 0]:
            file.write(str(count))
            file.write(',')
            file.write(str(value))
            file.write('\n')
            count += 1
