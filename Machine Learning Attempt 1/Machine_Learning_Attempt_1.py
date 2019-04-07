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
from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import Normalizer
from sklearn.neighbors import LSHForest
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

clf_names = [#'Multi-layer perceptron', 
             'Schochastic gradient descent', 
             #'Source vector machine', 
             'Multinomial naive Bayes', 
             'K-nearest neighbors', 
             'Decision tree']

clfs = [#MLPClassifier(warm_start = True, early_stopping = True), 
        SGDClassifier(tol = 1e-3, n_jobs = 16),
        #SVC(probability = True),
        MultinomialNB(),
        KNeighborsClassifier(algorithm = 'auto', n_jobs = 16),
        DecisionTreeClassifier()]

params = [#{'max_iter' : [500, 750, 1000]}, 
          {'loss' : ('log', 'modified_huber'), 
           'penalty' : ('l1', 'l2', 'elasticnet'), 
           'alpha' : [1e-1, 1e-2, 1e-3, 1e-4]},
          #{'C' : [1, 10, 100, 1000], 
           #'kernel' : ('linear', 'poly', 'rbf', 'sigmoid'), 
           #'degree' : [2, 3, 4, 5], 
           #'gamma' : ('auto', 'scale')},
          {'alpha' : [0.25, 0.5, 0.75, 1.0]},
          {'n_neighbors' : [5, 10, 15]},
          {'criterion' : ('gini', 'entropy'),
           'max_depth' : [5, 10, 15, 20, None]}]

#clfs = [SGDClassifier(tol = 1e-3, n_jobs = -1), MultinomialNB()]
#params = [{'loss' : ('log', 'modified_huber'), 
#           'penalty' : ('l1', 'l2', 'elasticnet'), 
#           'alpha' : [1e-1, 1e-2, 1e-3, 1e-4]},
#          {'alpha' : [0.25, 0.5, 0.75, 1.0]}]
results = []
for index, X_set, y_set in ezip(X_trains, y_trains):
    temp_results = []
    print(index + 1, " : ", words[index])
    vectorizer = TfidfVectorizer(analyzer = 'word', stop_words = list(words[index]), ngram_range = (2,2))
    X_set_I = vectorizer.fit_transform(X_set[:1000], y_set[:1000])
    X_test_set_I = vectorizer.transform(X_tests[index])
    transformer_train = Normalizer().fit(X_set_I)
    transformer_test = Normalizer().fit(X_test_set_I)
    X_set_I = transformer_train.transform(X_set_I)
    X_test_set_I = transformer_test.transform(X_test_set_I)
    for clfindex, classifier in enumerate(clfs):
        print(clf_names[clfindex], ": grid search.")
        grid = GridSearchCV(classifier, params[clfindex], cv = 5, iid = False)
        grid.fit(X_set_I, y_set[:1000])
        print(clf_names[clfindex], ": bagging.")
        best = BaggingClassifier(grid.best_estimator_, n_estimators = 100, max_samples = 0.5, max_features = 0.5)
        best.fit(X_set_I, y_set[:1000])
        temp_results.append(best.predict_proba(X_test_set_I))
    results.append(array_average_in_place(temp_results))
    #[results.append(k) for k in array_average_in_place(temp_results)]
with open('output.csv', 'w+') as file:
    file.write('Id,Expected')
    for index, item in enumerate(results):
        for entry in item:
            file.write('\n')
            file.write(','.join(map(str, entry)))
    print(index)

'''
#working below

clf = BaggingClassifier(SGDClassifier(), n_estimators = 100, max_samples = 0.25, max_features = 0.25)
for index, X_set, y_set in ezip(X_trains, y_trains):
    vectorizer = TfidfVectorizer(analyzer = 'word', stop_words = list(words[index]), ngram_range = (2,2))
    X_set_I = vectorizer.fit_transform(X_set, y_set)
    X_test_set_I = vectorizer.transform(X_tests[index])
    clf.fit(X_set_I, y_set)
    print(clf.predict_proba(X_test_set_I))
'''