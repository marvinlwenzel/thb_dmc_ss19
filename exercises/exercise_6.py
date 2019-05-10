import numpy as np
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
from csv import reader


def main():
    zoo_int = np.genfromtxt("data/zoo_numeric.csv", delimiter=" ",
                            dtype=int,
                            usecols=list(range(0, 17)),
                            encoding="iso-8859-1")
    zoo_int_labels = np.genfromtxt("data/zoo_numeric.csv", delimiter=" ",
                                   dtype=int,
                                   usecols=[17],
                                   encoding="iso-8859-1")
    clf = svm.SVC(kernel="linear")
    clf.fit(zoo_int, zoo_int_labels)

    print("Score total: {}".format(clf.score(zoo_int, zoo_int_labels)))

    print("Cross Val: {}".format(cross_val_score(clf, zoo_int, zoo_int_labels, cv=3)))

    train_size = 70

    my_train = zoo_int[0:train_size]
    my_train_labels = zoo_int_labels[0:train_size]

    my_test = zoo_int[train_size:]
    my_test_labels = zoo_int_labels[train_size:]

    my_clf = svm.SVC(kernel="linear")
    my_clf.fit(my_train, my_train_labels)

    my_test_predicted = my_clf.predict(my_test)

    mapping = whatever()
    names = [0] * len(mapping)
    for key in mapping:
        names[key] = mapping[key]

    print("Confusion Matrix: {}".format(confusion_matrix(my_test_labels, my_test_predicted)))
    print("Classification report:\n{}".format(classification_report(my_test_labels, my_test_predicted, target_names=names)))
    pass


def whatever():
    names = []
    with open("data/zoo_german.csv", "r", encoding="iso-8859-1") as zf:
        csv_reader = reader(zf, delimiter=",")
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
                line_count += 1
            else:
                names.append(row[len(row) - 1])
                line_count += 1
    zoo_int = np.genfromtxt("data/zoo_numeric.csv", delimiter=" ",
                            usecols=[17],
                            dtype=int,
                            encoding="iso-8859-1")
    mapping = dict()
    for i in range(0, len(names)):
        mapping[zoo_int[i]] = names[i]
    return mapping


if __name__ == '__main__':
    main()
