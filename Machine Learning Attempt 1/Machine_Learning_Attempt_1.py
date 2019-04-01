print("Importing modules.")
from sklearn.multiclass import OneVsOneClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
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

print("Formatting test set.")
pattern = re.compile(' ?\{([^\|]+)\|([^\}]+)\} ?')
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

'''
params_to_try = ['', {'n_neighbors' : [2, 4, 6, 8, 10], }, '', {'C' : [0.5, 2.1, 0.5]}, {'max_depth' : [5, 10, 15]}]

output_arrays = []
results = []
#clf = SGDClassifier(loss = 'log', penalty = 'l2', alpha = 1e-3, random_state = 42, max_iter = 15, tol = 1e-3)
clf = [MultinomialNB(), KNeighborsClassifier(), MLPClassifier(), SVC(), DecisionTreeClassifier()]
def test_c(clf, X_sets, y_sets):
    for index, X_set, y_set in ezip(X_sets, y_sets):
        vectorizer = TfidfVectorizer(analyzer = 'word', stop_words = list(words[index]), ngram_range = (2,2))
        #X_train, X_valid, y_train, y_valid = train_test_split(X_set, y_set)
        #X_train_tfidf = vectorizer.fit_transform(X_train, y_train)
        #X_valid_tfidf = vectorizer.transform(X_valid, y_valid)
        X_set_tfidf = vectorizer.fit_transform(X_set)
        y_set_tfidf = vectorizer.transform(y_set)
        for c in clf:
            print(clf[1], params_to_try[1])
            grid = GridSearchCV(clf[2], params_to_try[2], cv = 5)
            print(getnnz(X_set_tfidf), getnnz(y_set_tfidf))
            clf[0].fit(X_set_tfidf, y_set_tfidf)
            #grid.fit(X_set_tfidf, y_set_tfidf)
            #params = grid.best_params_
            pred = c.set_params(**params).fit(X_set, y_set).predict(X_tests)
            output_arrays.append(pred)
'''

X_trains, y_trains = format_trains(train_input)


clf = SGDClassifier(loss = 'log', penalty = 'l2', alpha = 1e-3, random_state = 42, max_iter = 15, tol = 1e-3)
for index, X_set, y_set in ezip(X_trains, y_trains):
    vectorizer = TfidfVectorizer(analyzer = 'word', stop_words = list(words[index]), ngram_range = (2,2))
    X_set_I = vectorizer.fit_transform(X_set, y_set)
    print(len(X_tests))
    print(len(X_tests[0]))
    X_test_set_I = vectorizer.transform(X_tests[index])
    clf.fit(X_set_I, y_set)
    print(clf.predict_proba(X_test_set_I))

#test_c(clf, X_trains, y_trains)

'''
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
'''