print("Importing modules.")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
import math
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
    count = 0
    with open(file, encoding = "utf-8 ") as f:
        for line in f:
                input.append(line.strip())
                count += 1
                if count > 15000:
                    pass
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

def crossentropy(i, j):
    summation = 0
    for x, y in zip(i[:,0], j):
        summation += y * math.log(x) + (1-y)*math.log(1-x)
    return -summation / len(i)

X_trains, y_trains = format_trains(train_input)
results = []
count = 1

clf = MultinomialNB()
bag = BaggingClassifier(clf, n_estimators = 100)
ce = 0
for index, X_set, y_set in ezip(X_trains, y_trains):
    print(words[index])
    print('vectorizing')
    X_set_train, X_set_val, y_set_train, y_set_val = train_test_split(X_set, y_set)
    vectorizer = TfidfVectorizer(analyzer = 'word', stop_words = list(words[index]), ngram_range = (1,3), min_df = 0.05, max_df = 0.95)
    X_set_train_I = vectorizer.fit_transform(X_set_train, X_set_val)
    X_set_val_I = vectorizer.transform(X_set_val)
    X_test_set_I = vectorizer.transform(X_tests[index])
    print('fitting')
    bag.fit(X_set_train_I, y_set_train)
    prediction = bag.predict_proba(X_set_val_I)
    results.append(bag.predict_proba(X_test_set_I))
    y_wordmatrix = [1 if x == words[index][0] else 0 for x in y_set_val]
    ce += crossentropy(prediction, y_wordmatrix)
print(ce / 25)
with open('output.csv', 'w', encoding = 'utf-8') as file:
    file.write('Id,Expected')
    for result in results:
        for line in result:
            file.write('\n{},{}'.format(count, line[0]))
            count += 1