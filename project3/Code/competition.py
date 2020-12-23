import numpy as np
from sklearn import svm
from sklearn.preprocessing import normalize
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB,ComplementNB
from sklearn.linear_model import LogisticRegression

import csv

def readFile(filename):
    return np.genfromtxt(filename, delimiter=',')

def writeFile(filename,labels):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['id','label'])
        for i, label in enumerate(labels):
            writer.writerow([i,label])

def data_normalization(features):
    return normalize(features, axis=0, norm='l2')

def support_vector_machine_find_parameter(training_set,testing_set,training_labels,c_value,gamma):            # To test and get best parameter and will use for later real predict
    TP, FN, FP, TN = 0, 0, 0, 0
    a, p, r, f = [], [], [], []

    training_set,testing_set = data_normalization(training_set),data_normalization(testing_set)
    clf = svm.SVC(gamma='auto',kernel='linear',C= c_value)
    # clf.fit(X=training_set,y=training_labels)
    # a = clf.predict(X=testing_set)

    kf = KFold(n_splits=10)
    kf.get_n_splits(training_set)

    for train_index, test_index in kf.split(training_set):
        clf.fit(training_set[train_index],training_labels[train_index])
        result = clf.predict(training_set[test_index])

        for i, idx in enumerate(test_index):
            if result[i] == 1 and training_labels[idx] == 1:
                    TP += 1
            elif result[i] == 0 and training_labels[idx] == 1:
                    FN += 1
            elif result[i] == 1 and training_labels[idx] == 0:
                    FP += 1
            elif result[i] == 0 and training_labels[idx] == 0:
                    TN += 1

        accuracy = (TP + TN) / (TP + TN + FP + FN)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        F_measure = 2 * recall * precision / (recall + precision)

        a.append(accuracy)
        p.append(precision)
        r.append(recall)
        f.append(F_measure)

    print("C_value : " + str(c_value))
    print("Accuracy : " + np.str(np.mean(a)))
    print("Precision : " + np.str(np.mean(p)))
    print("Recall : " + np.str(np.mean(r)))
    print("F_measure : " + np.str(np.mean(f)))
    print("_____________________________________________________________")

def knn_find_parameter(training_set,testing_set,training_labels,k):                             # To get best parameter k and will be used in later real predicting
    TP, FN, FP, TN = 0, 0, 0, 0
    a, p, r, f = [], [], [], []

    training_set, testing_set = data_normalization(training_set), data_normalization(testing_set)
    kf = KFold(n_splits=10)
    kf.get_n_splits(training_set)

    for train_index, test_index in kf.split(training_set):
        n = KNeighborsClassifier(n_neighbors=k)
        n.fit(training_set[train_index], training_labels[train_index])
        result = n.predict(training_set[test_index])
        for i, idx in enumerate(test_index):
            if result[i] == 1 and training_labels[idx] == 1:
                    TP += 1
            elif result[i] == 0 and training_labels[idx] == 1:
                    FN += 1
            elif result[i] == 1 and training_labels[idx] == 0:
                    FP += 1
            elif result[i] == 0 and training_labels[idx] == 0:
                    TN += 1

        accuracy = (TP + TN) / (TP + TN + FP + FN)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        F_measure = 2 * recall * precision / (recall + precision)

        a.append(accuracy)
        p.append(precision)
        r.append(recall)
        f.append(F_measure)

    print("K_value : " + str(k))
    print("Accuracy : " + np.str(np.mean(a)))
    print("Precision : " + np.str(np.mean(p)))
    print("Recall : " + np.str(np.mean(r)))
    print("F_measure : " + np.str(np.mean(f)))
    print("_____________________________________________________________")

def dection_tree_find_parameter(training_set,testing_set,training_labels,calculate,depth):
    TP, FN, FP, TN = 0, 0, 0, 0
    a, p, r, f = [], [], [], []

    training_set, testing_set = data_normalization(training_set), data_normalization(testing_set)
    kf = KFold(n_splits=10)
    kf.get_n_splits(training_set)

    for train_index, test_index in kf.split(training_set):

        clf = tree.DecisionTreeClassifier(criterion=calculate,max_depth=depth)
        clf = clf.fit(training_set[train_index], training_labels[train_index])
        result = clf.predict(training_set[test_index])

        for i, idx in enumerate(test_index):
            if result[i] == 1 and training_labels[idx] == 1:
                TP += 1
            elif result[i] == 0 and training_labels[idx] == 1:
                FN += 1
            elif result[i] == 1 and training_labels[idx] == 0:
                FP += 1
            elif result[i] == 0 and training_labels[idx] == 0:
                TN += 1

        accuracy = (TP + TN) / (TP + TN + FP + FN)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        F_measure = 2 * recall * precision / (recall + precision)

        a.append(accuracy)
        p.append(precision)
        r.append(recall)
        f.append(F_measure)

    print("criterion : " + str(calculate) + "   " + "Max_depth : " + str(depth))
    print("Accuracy : " + np.str(np.mean(a)))
    print("Precision : " + np.str(np.mean(p)))
    print("Recall : " + np.str(np.mean(r)))
    print("F_measure : " + np.str(np.mean(f)))
    print("_____________________________________________________________")

def naive_bayes_find_parameter(training_Set,testing_set,training_labels):
    TP, FN, FP, TN = 0, 0, 0, 0
    a, p, r, f = [], [], [], []

    # training_set, testing_set = data_normalization(training_set), data_normalization(testing_set)
    kf = KFold(n_splits=10)
    kf.get_n_splits(training_set)

    for train_index, test_index in kf.split(training_set):

        gnb = GaussianNB()
        result = gnb.fit(training_set[train_index], training_labels[train_index]).predict(training_set[test_index])

        for i, idx in enumerate(test_index):
            if result[i] == 1 and training_labels[idx] == 1:
                TP += 1
            elif result[i] == 0 and training_labels[idx] == 1:
                FN += 1
            elif result[i] == 1 and training_labels[idx] == 0:
                FP += 1
            elif result[i] == 0 and training_labels[idx] == 0:
                TN += 1

        accuracy = (TP + TN) / (TP + TN + FP + FN)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        F_measure = 2 * recall * precision / (recall + precision)

        a.append(accuracy)
        p.append(precision)
        r.append(recall)
        f.append(F_measure)

    print("Accuracy : " + np.str(np.mean(a)))
    print("Precision : " + np.str(np.mean(p)))
    print("Recall : " + np.str(np.mean(r)))
    print("F_measure : " + np.str(np.mean(f)))
    print("_____________________________________________________________")

def logical_regression_parameter(training_Set,testing_set,training_labels,penalty,iter):
    TP, FN, FP, TN = 0, 0, 0, 0
    a, p, r, f = [], [], [], []

    # training_set, testing_set = data_normalization(training_set), data_normalization(testing_set)
    kf = KFold(n_splits=10)
    kf.get_n_splits(training_set)

    for train_index, test_index in kf.split(training_set):

        clf = LogisticRegression(random_state=0,penalty=penalty,max_iter=iter).fit(training_set[train_index], training_labels[train_index])
        result = clf.predict(training_set[test_index])

        for i, idx in enumerate(test_index):
            if result[i] == 1 and training_labels[idx] == 1:
                TP += 1
            elif result[i] == 0 and training_labels[idx] == 1:
                FN += 1
            elif result[i] == 1 and training_labels[idx] == 0:
                FP += 1
            elif result[i] == 0 and training_labels[idx] == 0:
                TN += 1

        accuracy = (TP + TN) / (TP + TN + FP + FN)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        F_measure = 2 * recall * precision / (recall + precision)

        a.append(accuracy)
        p.append(precision)
        r.append(recall)
        f.append(F_measure)
    print("Penalty : " + penalty + "       Max iter : "+str(iter))
    print("Accuracy : " + np.str(np.mean(a)))
    print("Precision : " + np.str(np.mean(p)))
    print("Recall : " + np.str(np.mean(r)))
    print("F_measure : " + np.str(np.mean(f)))
    print("_____________________________________________________________")

def SupportVectorMachine(training_set,testing_set,training_labels):
    training_set, testing_set = data_normalization(training_set), data_normalization(testing_set)
    clf = svm.SVC(gamma=1e-8, kernel='linear', C=3.5)
    clf.fit(X=training_set, y=training_labels)
    return clf.predict(X=testing_set)

def knn(training_set,testing_set,training_labels):
    training_set, testing_set = data_normalization(training_set), data_normalization(testing_set)
    n = KNeighborsClassifier(n_neighbors=3)
    n.fit(training_set, training_labels)
    labels = n.predict(testing_set)
    return labels

def decision_tree(training_set,testing_set,training_labels):
    clf = tree.DecisionTreeClassifier(criterion='entropy',max_depth=12)
    clf = clf.fit(training_set, training_labels)
    labels = clf.predict(testing_set)
    # print(np.count_nonzero(labels==0))
    return labels

def naive_bayes(training_set,testing_set,training_labels):
    # train,test = load_iris(return_X_y=True)
    gnb = GaussianNB()
    labels = gnb.fit(training_set, training_labels).predict(testing_set)
    # print(np.count_nonzero(labels == 0))
    return labels

def logical_regression(training_set,testing_set,training_labels):
    clf = LogisticRegression(random_state=0).fit(training_set, training_labels)
    labels = clf.predict(testing_set)
    # print(np.count_nonzero(labels==0))
    return labels

if __name__ == "__main__":
    training_set = readFile('train_features.csv')
    testing_set = readFile('test_features.csv')
    training_labels = np.delete(readFile('train_labels.csv'),0,axis=0)

    # c = [1,1.5,2,2.5,3,3.5,4]                                                                     # Select best c and gamma for svm , got 3.5, 1e-8
    # gamma = [1e-1,5e-1,1e-2,5e-2,1e-3,5e-3,1e-4,5e-4,1e-5,5e-5,1e-6,5e-6,1e-7,5e-7,1e-8,5e-8]     # F_measure = 0.8990667620666641
    # for i in c:
    #     for x in gamma:
    #         support_vector_machine_find_parameter(training_set[:,1:],testing_set[:,1:],training_labels[:,1],i,x)

    # k = [2,3,4,5,6,7,8,9,10]                                                                      # Select best k for knn, got 3
    # for i in k:                                                                                   # F_measure = 0.8931751550271458
    #     knn_find_parameter(training_set[:,1:],testing_set[:,1:],training_labels[:,1], i)

    # cal = ["gini", "entropy"]                                                                 # Select Criterion and Max_depth, we got Entropy, and 120
    # dep = [4,5,6,7,8,9,10,11,12,13,14]                                                        # F = 0.8332889402922363
    # for i in cal:
    #     for x in dep:
    #         dection_tree_find_parameter(training_set[:,1:],testing_set[:,1:],training_labels[:,1], i,x)

    # naive_bayes_find_parameter(training_set[:,1:],testing_set[:,1:],training_labels[:,1])       # Using Gaussian NB      F = 0.6298112685956598

    # penaty = ['l1','l2']                                                                            # Using l1, and Max iter 20   F = 0.8685169407880642
    # iter = [20,30,40,50,60,70,80,90,100]
    # for x in penaty:
    #     for i in iter:
    #         logical_regression_parameter(training_set[:,1:],testing_set[:,1:],training_labels[:,1], x, i)

    kf = KFold(n_splits=10)
    kf.get_n_splits(training_set)

    result = {0: 0, 1: 0}
    arr = []
    labels = []
    for train_index, test_index in kf.split(training_set):
        t = training_set[train_index]
        l = training_labels[train_index]
        svm_labels = SupportVectorMachine(t[:,1:],testing_set[:,1:],l[:,1])
        knn_labels = knn(t[:,1:],testing_set[:,1:],l[:,1])
        dt_labels = decision_tree(t[:,1:],testing_set[:,1:],l[:,1])
        nb_labels = naive_bayes(t[:,1:],testing_set[:,1:],l[:,1])
        lr_labels = logical_regression(t[:,1:],testing_set[:,1:],l[:,1])


        for i in range(testing_set.shape[0]):
            result = {0: 0, 1: 0}
            if(len(arr)<=i):
                arr.append([])
            result[svm_labels[i]] += 0.8990667620666641
            result[knn_labels[i]] += 0.8931751550271458
            result[dt_labels[i]] += 0.8931751550271458
            result[nb_labels[i]] += 0.6298112685956598
            result[lr_labels[i]] += 0.8685169407880642
            arr[i].append(result[0])
            arr[i].append(result[1])

    for idx, row in enumerate(arr):
        zero, one = 0,0
        for i, value in enumerate(row):
            if i % 2 == 0:
                zero += value
            else:
                one += value
        labels.append(0) if zero > one else labels.append(1)


    writeFile('submission.csv',labels)
