from GeneralTensorCalculator import GeneralTensorCalculator
import numpy as np


# Examples #
def my_test_f(arguments: list) -> float:
    result = 1

    for a in arguments:
        result *= a

    return result


def my_test_f_2(arguments: list):
    result = 1

    for a in arguments:
        result = np.kron(result, a)

    return result


gtc = GeneralTensorCalculator(my_test_f)
args1 = [[1, 2, 3], [40, 50], [600, 700, 800]]
t = gtc.calculate_tensor(args1)

sigma_z = np.array([[1, 0], [0, -1]])

gtc2 = GeneralTensorCalculator(my_test_f_2)
t2 = gtc2.calculate_tensor([
    [1*sigma_z, 2*sigma_z],
    [2*np.eye(4), 3*np.eye(4), 4*np.eye(4),5*np.eye(5)],
    [10*sigma_z, 1000*sigma_z, 10e6*sigma_z]
])

t3 = gtc.calculate_tensor_to_increasing_list(args1)
print(t3)
t3.sort()
print(t3)
