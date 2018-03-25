import numpy as np


# <np.asarray> :transfer value in (arr or mat) to otherValueType
def use_np_array():
    arr = [[1, 2], [3.2, 4]]
    a = np.asarray(arr, "float32") / 5
    return a


def use_np_empty():
    arr = np.empty([2, 2], dtype=int)
    arr[0] = [1]

    return arr

# blog about np.array :(https://blog.csdn.net/baoqian1993/article/details/51725140)

def use_array_and_ndarray():
    data1 = [[1.2, 1.3], [2.2, 2.3]]
    arr1 = np.array(data1)
    print(arr1.shape)
    print(arr1.dtype)
    arr2 = np.zeros([2,3])
    arr3 = np.empty([3,2])
    arr4 = np.arange(2,6,0.5)
    print(arr4)
    arr5 = np.ones((3,5)) * 2
    print(arr5)
    arr5.astype(float)  #transfer dtype to float
    print(arr5/5)

    #slicing for 1_d array
    arr6 = arr4
    print(arr6[3:6])
    #slicing for 2d array
    arr7 = arr5[0:3, 0:3]
    print(arr7)
    # flatten: transfer ndArray to 1d_array
    arr8 = np.ndarray.flatten(arr7)
    print(arr8)

    arr9 = arr8.reshape((3,3))
    print(arr9)

def use_tf_argmax():
    data = [[1,3,4,2], [2,1,0,9]]
    arr = np.array(data)
    # argmax(arr, 0)  search based on column
    print(np.argmax(arr,0))

    # argmax(arr, 1)  search based on row
    print(np.argmax(arr, 1))




if __name__ == "__main__":
    # print(use_np_array())
    # print(use_np_empty())
    # use_array_and_ndarray()
    use_tf_argmax()