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

def array_average_in_place(lst):
    result = []
    for row in range(len(lst[0])):
        temp = []
        sum = 0
        for array in lst:
            #sum += array.item(row, 0)
            temp.append(array.item(row, 0))
        result.append(temp)
        #result.append(sum/len(lst))
    print(len(result))
    return result

X_trains, y_trains = format_trains(train_input)

results = []
count = 1
scalar = StandardScaler(with_mean = False)
clf1 = MultinomialNB()
select1 = RFECV(clf1, cv = 5)
bag = BaggingClassifier(clf1, n_estimators = 500, max_samples = 1.0, max_features = 1.0)
for index, X_set, y_set in ezip(X_trains, y_trains):
    print(words[index])
    print('vectorizing')
    vectorizer = TfidfVectorizer(analyzer = 'word', stop_words = list(words[index]), ngram_range = (1,3), min_df = 0.05, max_df = 0.95)
    X_set_I = vectorizer.fit_transform(X_set, y_set)
    X_test_set_I = vectorizer.transform(X_tests[index])
    print('scaling')
    X_set_I = scalar.fit_transform(X_set_I)
    X_test_set_I = scalar.transform(X_test_set_I)
    print('fitting MultinomialNB')
    clf1.fit(X_set_I, y_set)
    print('filtering MultinomialNB')
    select1.fit(X_set_I, y_set)
    print('filtering DecisionTree')
    #select2.fit(X_set_I, y_set)
    #bag.fit(X_set_I, y_set)
    #ada.fit(X_set_I, y_set)
    print(clf1.predict_proba(X_test_set_I))
    results.append(clf1.predict_proba(X_test_set_I))
with open('output2.csv', 'w', encoding = 'utf-8') as file:
    file.write('Id,Expected')
    for result in results:
        for line in result:
            file.write('\n{},{}'.format(count, line[0]))
            count += 1

#working below
'''
clf = BaggingClassifier(SGDClassifier(), n_estimators = 100, max_samples = 0.25, max_features = 0.25)
for index, X_set, y_set in ezip(X_trains, y_trains):
    vectorizer = TfidfVectorizer(analyzer = 'word', stop_words = list(words[index]), ngram_range = (2,2))
    X_set_I = vectorizer.fit_transform(X_set, y_set)
    X_test_set_I = vectorizer.transform(X_tests[index])
    clf.fit(X_set_I, y_set)
    print(clf.predict_proba(X_test_set_I))
'''