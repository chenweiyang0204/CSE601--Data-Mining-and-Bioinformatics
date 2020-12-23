import numpy as np
from collections import Counter

def readFile(filename):
    contents = open(filename, 'r')
    firstLine = contents.readline().split()
    contents.close()
    num = [i for i in range(0, len(firstLine))]

    features = np.genfromtxt(filename, usecols=num, dtype='str')

    return features

def n_folds_cross_validation(n_samples, n_folds):
    number_each_fold = np.round(n_samples/n_folds).astype(int)

    retArr = []
    index = 0
    for i in range(n_folds):
        if(i!=n_folds - 1):
            retArr.append([x for x in range(index,(index+number_each_fold))])
            index += number_each_fold
        else:
            retArr.append([x for x in range(index,n_samples)])
    return retArr

def data_normalization(features):
    features = features.T
    normalized_features = []

    for idx, feature in enumerate(features[:-1]):
        if feature[0].replace(".","").isdigit():
            min_v , max_v = np.min(feature.astype(float)), np.max(feature.astype(float))
            col = np.asarray([(float(i) - min_v) / (max_v - min_v) for i in feature]).astype(str)
            normalized_features.append(col)
        else:
            normalized_features.append(feature)
    normalized_features.append((features[-1,:]))

    return np.asarray(normalized_features).T

def euclidean_distance(x,y,diff):
    dis = 0
    for i, item in enumerate(x[:-1]):
        if(item.replace(".","").isdigit()):
            dis += (float(item) - float(y[i]))**2
        else:
            if(item != y[i]):
                dis += diff
    return dis**0.5

def voting(idx,training_set):
    c = {'0':0, '1':0}
    for i in idx :
        c[training_set[i,-1]] += 1
    if(c['0'] > c['1']):
        return '0'
    elif c['0'] < c['1']:
        return '1'
    else:
        return training_set[idx[0],-1]

def KNN(training_set, testing_set, K, diff):
    TP, FN, FP, TN = 0, 0, 0, 0

    for test in testing_set:
        records = []
        for train in training_set:
            records.append(euclidean_distance(test,train,diff))
        idx = np.argsort(records)[:K]

        label = voting(idx,training_set)

        if int(test[-1]) == 1 and int(label) == 1:
            TP += 1
        elif int(test[-1]) == 1 and int(label) == 0:
            FN += 1
        elif int(test[-1]) == 0 and int(label) == 1:
            FP += 1
        elif int(test[-1]) == 0 and int(label) == 0:
            TN += 1
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F_measure = 2 * recall * precision / (recall + precision)

    return accuracy, precision, recall, F_measure

def running(features, folds_arr,diff ,K):
    all_fold = [x for x in range(features.shape[0])]
    result = []
    for test_fold in folds_arr:
        train_fold = np.setdiff1d(all_fold, test_fold)

        testing_set, training_set = np.asarray([features[i] for i in test_fold]), np.asarray(
            [features[i] for i in train_fold])

        result.append(KNN(training_set, testing_set, K, diff))

    result = np.asarray(result)

    print("Accuracy : " + np.str(np.mean(result[:, 0])))
    print("Precision : " + np.str(np.mean(result[:, 1])))
    print("Recall : " + np.str(np.mean(result[:, 2])))
    print("F_measure : " + np.str(np.mean(result[:, 3])))

if __name__ == "__main__":
    filename = 'project3_dataset2.txt'

    features = readFile(filename)
    features = data_normalization(features)

    rows, cols = features.shape

    folds_arr = n_folds_cross_validation(rows, 10)

    running(features, folds_arr, K = 10,diff= 1)

    #############  DEMO  ##########################

    # training = readFile(filename='project3_dataset3_train.txt')
    # testing = readFile(filename='project3_dataset3_test.txt')
    # accuracy , precision, recall,F_measure = KNN(training_set=training,testing_set=testing,K=9,diff =1)
    #
    # print("Accuracy : " + str(accuracy))
    # print("Precision : " + str(precision))
    # print("Recall : " + str(recall))
    # print("F_measure : " + str(F_measure))