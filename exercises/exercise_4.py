import numpy
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression


def scale_normalize(input_array):
    min = input_array.min
    max = input_array.max

    result = numpy.copy(input_array)
    for i in range(len(input_array)):
        result[i] = (input_array[i] - min) / (max - min)

    return result


def two():
    input = numpy.genfromtxt('data/two.csv')
    plt.scatter(input[:, 0], input[:, 1])
    plt.show()

    input_a = input[:, 0]
    input_b = input[:, 1]

    min_a = input_a.min()
    max_a = input_a.max()
    mean_a = input_a.mean()
    var_a = input_a.var()

    min_b = input_b.min()
    max_b = input_b.max()
    mean_b = input_b.mean()
    var_b = input_a.var()

    print("min: {}, max: {}, mean: {}, var: {}".format(min_a, max_a, mean_a, var_a))
    print("min: {}, max: {}, mean: {}, var: {}".format(min_b, max_b, mean_b, var_b))

    coeff = numpy.corrcoef(input_a, input_b)

    print("Corrcoef")
    print(coeff)

    scaler = MinMaxScaler()
    scaled_a = scaler.fit_transform(input_a.reshape(-1, 1))

    scaler = MinMaxScaler()
    scaled_b = scaler.fit_transform(input_b.reshape(-1, 1))

    scaled_coeff = numpy.corrcoef(scaled_a[:, 0], scaled_b[:, 0])

    print("MinMaxed Coef")
    print(scaled_coeff)

    std_a = StandardScaler().fit_transform(input_a.reshape(-1, 1))
    std_b = StandardScaler().fit_transform(input_b.reshape(-1, 1))

    std_coeff = numpy.corrcoef(std_a[:, 0], std_b[:, 0])

    print("Std Coef")
    print(std_coeff)

    linReg = LinearRegression()
    linReg.fit(input_a.reshape(-1, 1), input_b.reshape(-1, 1))

    print("LinReg: {}".format(linReg.coef_))


if __name__ == '__main__':
    two()