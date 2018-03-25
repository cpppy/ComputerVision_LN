import numpy as np


#transfer value in (arr or mat) to otherValueType
def use_np_array():
    arr = [[0, 2], [3.2, 4]]
    a = np.asarray(arr, "float32")
    return a

if __name__ == "__main__":
    print(use_np_array())