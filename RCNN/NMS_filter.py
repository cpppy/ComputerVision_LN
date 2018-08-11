import numpy as np
import preprocessing_RCNN as prep


def filterByNMS(rectArr, probaArr):
    print("using NMS filter on predict results by svm.")
    checkedIndex = []
    indexArr = np.argsort(probaArr).tolist()
    indexArr.reverse()
    print("index of probaArr by descending :", indexArr)
    resRectArr = []
    resProbArr = []

    tmpIndexArr = indexArr.copy()
    while(len(indexArr) > 0):
        print(" *** cycle in NMS *** ")
        indexOfMaxProba = indexArr[0]
        checkedIndex.append(indexOfMaxProba)
        resRectArr.append(rectArr[indexOfMaxProba])
        resProbArr.append(probaArr[indexOfMaxProba])
        tmpIndexArr.remove(indexOfMaxProba)
        for index in indexArr:
            print(" - index in NMS :", index)
            if (index == indexOfMaxProba):
                continue
            iouValue = prep.calcIOUForSameRectStructureInput(rectArr[indexOfMaxProba], rectArr[index])
            print("iou value :", iouValue)
            if (iouValue > 0.6):
                # drop this childImg
                tmpIndexArr.remove(index)
                print("delete index from Arr, delIndex :", index)
        indexArr = tmpIndexArr
    print("do NMS over, resRectArr :", resRectArr)
    return resRectArr, resProbArr, checkedIndex