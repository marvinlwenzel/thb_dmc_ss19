import numpy as np
import matplotlib.pyplot as plt
import sklearn.feature_selection as fs
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import math


def main():
    two = np.genfromtxt('data/two.csv')
    two = StandardScaler().fit_transform(two)
    pca = PCA()
    pca.fit(two)

    base_corr = np.corrcoef(two.transpose())

    print(
        "Q1a)\nVariance weights:\nÂ1: {}\nÂ2: {}".format(pca.explained_variance_ratio_[0],
                                                         pca.explained_variance_ratio_[1]))

    two = pca.transform(two)

    pcaed = pca.transform(two)

    pcaed_corr = np.corrcoef(pcaed.transpose())

    print("Q1b)\nrotated by {}°".format(np.degrees(math.acos(pca.components_[0, 0]))))
    print("Q1c)\nOg:\n{}\nPCA:\n{}".format(base_corr, pcaed_corr))

    reduced = pcaed[:, 0]

    zs = np.zeros((len(reduced)))
    x = np.array([reduced, zs]).transpose()
    red_inverse = pca.inverse_transform(x)

    plt.scatter(two[:, 0], two[:, 1])
    plt.scatter(pcaed[:, 0], pcaed[:, 1])
    plt.scatter(red_inverse[:, 0], red_inverse[:, 1])
    plt.show()

    print("Q2a)\nBinäre werden zu 0 und 1")

    zoo = np.genfromtxt("data/zoo_german.csv", delimiter=",",
                        skip_header=1,
                        usecols=list(range(1, 13)) + list(range(15, 18)),
                        dtype=bool,
                        encoding="iso-8859-1")

    print("Q2b)\nLädt csv mit utf8 statt iso88591 mit Komma Deleimiter, "
          "nutz nur zeilen 1-12,15-16, skipt header zeile und setzt datentypen zu boolean")

    selector = fs.VarianceThreshold()
    selector.fit_transform(zoo)

    t = np.sort(selector.variances_)[2]
    selector = fs.VarianceThreshold(threshold=t)

    selected = selector.fit_transform(zoo)

    print("Q2c)\nOg shape: {}\nSelectedShape: {}".format(zoo.shape, selected.shape))

    holes = np.genfromtxt('data/two_missing.csv', delimiter=",")

    averages = np.nanmean(holes, axis=0)

    fixed = np.copy(holes)
    for i in range(0, holes.shape[0]):
        for j in range(0, holes.shape[1]):
            if math.isnan(holes[i,j]):
                fixed[i,j] = averages[j]

    fixed_corr = np.corrcoef(fixed.transpose())

    print("Q3)\nTwo Corr:\n{}\nHoles Corr:\n{}".format(base_corr, fixed_corr))

    print("Hi")


if __name__ == '__main__':
    main()
