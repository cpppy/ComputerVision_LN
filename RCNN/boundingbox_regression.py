from sklearn.linear_model import Ridge
import numpy as np
import os
import preprocessing_RCNN as prep
import selectivesearch
import tools
from sklearn.externals import joblib
import cv2
import math
import RCNN_output
import tflearn
from sklearn.linear_model import LogisticRegression


def load_regression_train_data(txtFilePath):
    print("load train data for bbox regression.")
    npyOutputPath = txtFilePath.rsplit('.', 1)[0].strip()
    if len(os.listdir(npyOutputPath)) == 0:
        print("reading %s's bbox regression dataset" % txtFilePath.split('\\')[-1])

        # generate region proposals for images in 1.txt/ 2.txt,
        # and store it into a npy file
        generateDataForRegression(txtFilePath,    # ./regression_train/1.txt
                                  npyOutputPath,  # ./regression_train/1
                                  threshold=0.6)
    print("restoring bbox regression dataset ...")
    images, correctionCoefs = prep.load_from_npy(npyOutputPath)
    print("imageArr as X, size :", np.shape(images))
    print("correctionCoefArr as Y, size :", np.shape(correctionCoefs))
    return images, correctionCoefs


def fittingBBoxRegression(bbrModelSavePath, modelOfCNN):
    print("training bbox regression model, begin.")
    regressionModelDict = {}
    files = os.listdir("./regression_train")
    for train_file in files:
        print("current fileName :", train_file)
        if train_file.split('.')[-1] == 'txt':
            txtFilePath = os.path.join("./regression_train", train_file)
            print("txt file path :", txtFilePath)
            X, Y = load_regression_train_data(txtFilePath)
            featureArr = []
            for index, image in enumerate(X):
                # extract features
                features = modelOfCNN.predict([image])
                # features output from alex_net(4096), as the input of regression model
                featureArr.append(features[0])
                tools.view_bar("extract features of %s" % train_file, index + 1, len(X))
            print(' ')
            print("feature for regression model dimension")
            print("size of featureArr :", np.shape(featureArr))
            # bbox regression training
            ridgeRegresModel = Ridge(alpha=1.0)

            # logRegressModel = LogisticRegression(class_weight='balanced',
            #                                      solver='lbfgs',
            #                                      multi_class='multinomial',
            #                                      verbose=1,
            #                                      n_jobs=-1,
            #                                      max_iter=1000)

            print("fit bbox regression.")
            ridgeRegresModel.fit(featureArr, Y)

            # use joblib package to store svm_training result
            print("finish fitting, dump to pickle file ... ")
            joblib.dump(ridgeRegresModel, os.path.join(bbrModelSavePath, train_file.split(".")[0] + "_bbox_regression.pkl"))
            objectClassTag = train_file.split('.')[0]
            regressionModelDict[objectClassTag] = ridgeRegresModel
    return regressionModelDict


def generateDataForRegression(dataInfoFilePath,
                              savePath,
                              threshold=0.6):
    print("function = generateDataForRegression, filePath : ", dataInfoFilePath)
    f = open(dataInfoFilePath, "r")
    fileInfoList = f.readlines()
    # random.shuffle(fileInfoList)
    print("line num of dataInfoFile : ", len(fileInfoList))
    lineIndex = 0
    for line in fileInfoList:
        print(str(lineIndex) + " ---> Current line :", line)
        # read image file pathInfo in this line
        infoArr = line.split(" ")
        imageFilePath = infoArr[0]
        objectClass = int(infoArr[1])
        objectBoxPositonInfo = infoArr[2]
        print("image path :", imageFilePath)
        print("object class :", objectClass)
        print("object position info str :", objectBoxPositonInfo)
        positionArrInString = objectBoxPositonInfo.split(",")
        notationRectArr = [int(s) for s in positionArrInString]
        # for s in positionArrInString:
        #     notationRectArr.append(int(s))
        print("object position arr :", notationRectArr)
        print("read imageFile info success...")

        # read img from filePath
        img = cv2.imread(imageFilePath)
        print("original img size: ", np.shape(img))
        # print(img)
        # scale : size of the smallest region proposals
        # sigma : Width of Gaussian kernel for felzenszwalb segmentation
        # min_size : min size of regions
        img_lbl, regions = selectivesearch.selective_search(img,
                                                            scale=500,
                                                            sigma=0.9,
                                                            min_size=10)
        # img_lbl ??????
        img_lbl_0 = img_lbl[0]
        regions_0 = regions[0]
        print("child img size :", np.shape(img_lbl_0))
        print(regions_0)  # {'rect': (0, 0, 499, 441), 'size': 140000, 'labels': [0.0]}

        # choose proposal regions
        proposalRegionCandiatesInRect = set()
        choosedChildImgArr = []
        choosedCorrectionCoefArr = []

        for childImageInfo in regions:
            # childImageInfo : {'rect': (0, 0, 499, 441), 'size': 140000, 'labels': [0.0]}
            if (proposalRegionCandiatesInRect.__contains__(childImageInfo['rect'])):
                print(" ------ childImage exist in candidates set, continue.")
                continue
            # delete child images which is too small
            childImageSize = childImageInfo['size']
            childImageRect = childImageInfo['rect']  # 'rect' : xStart, yStart, width, length
            if (childImageInfo['size'] < 220
                or childImageRect[2] * childImageRect[3] < 500):
                continue
            # crop original image by childImageRect
            childImg, childImgDetailRect = cropImage(img, childImageRect)
            childImgShape = np.shape(childImg)
            # check childImage
            if (len(childImg) == 0
                or childImgShape[0] == 0
                or childImgShape[1] == 0
                or childImgShape[2] == 0):
                continue
            # resize image
            resizedChildImg = resize_image(childImg, 224, 224)
            proposalRegionCandiatesInRect.add(childImageRect)
            resizedChildImgInFloat = np.asarray(resizedChildImg, dtype="float32")

            # calc IOU
            # use rect in notation to compare with generated rect by selective search
            iouValue = calcIOU(notationRectArr, childImgDetailRect)
            print("IOU :", iouValue)

            if (iouValue > threshold):
                # this childImg can be used as training data for bbox regression
                # calc correctionCoef
                currCoef = calcCorrectionCoef(childImageRect, notationRectArr)
                print("correction coef :", currCoef)
                # add this childImg to result
                choosedChildImgArr.append(resizedChildImgInFloat)
                choosedCorrectionCoefArr.append(currCoef)

        tools.view_bar("processing image of %s" % dataInfoFilePath.split('\\')[-1].strip(), lineIndex + 1,
                       len(fileInfoList))
        lineIndex += 1
        # save childImg collection to npy file
        originalImgFileName = imageFilePath.split('/')[-1]
        headOfOriginalImgFileName = originalImgFileName.split('.')[0].strip()
        tailOfGeneratedFileName = '_data.npy'
        generatedFilePath = os.path.join(savePath, headOfOriginalImgFileName) + tailOfGeneratedFileName
        np.save(generatedFilePath, [choosedChildImgArr, choosedCorrectionCoefArr])
        print("store bbox regression training data into file :", generatedFilePath)
    f.close()


def calcCorrectionCoef(rectOfProposal, rectOfNotation):
    correctionCoef = []

    rectOfProposal = [float(elem) for elem in rectOfProposal]
    rectOfNotation = [float(elem) for elem in rectOfNotation]

    Px = rectOfProposal[0]
    Py = rectOfProposal[1]
    Pw = rectOfProposal[2]
    Ph = rectOfProposal[3]

    Gx = rectOfNotation[0]
    Gy = rectOfNotation[1]
    Gw = rectOfNotation[2]
    Gh = rectOfNotation[3]

    tx = (Gx - Px) / Pw
    ty = (Gy - Py) / Ph
    tw = math.log(Gw / Pw)
    th = math.log(Gh / Ph)

    correctionCoef = np.array([tx, ty, tw, th])

    return correctionCoef


# Clip Image
def cropImage(originImage, rectArr):
    x0 = rectArr[0]
    y0 = rectArr[1]
    width = rectArr[2]
    hight = rectArr[3]
    x1 = x0 + width
    y1 = y0 + hight
    # error use : return img[x:x_1, y:y_1, :], [x, y, x_1, y_1, w, h]
    # actually, child image = original image[row0, row1, col0, col1]
    # but compare to the coordinate sys in math, row0 = y0, col0 = x0
    return originImage[y0:y1, x0:x1, :], [x0, y0, x1, y1, width, hight]


def resize_image(in_image, new_width, new_height, out_image=None, resize_mode=cv2.INTER_CUBIC):
    img = cv2.resize(in_image, (new_width, new_height), resize_mode)
    if out_image:
        cv2.imwrite(out_image, img)
    return img


# IOU Part 1
def if_intersection(xmin_a, xmax_a, ymin_a, ymax_a, xmin_b, xmax_b, ymin_b, ymax_b):
    if_intersect = False
    if xmin_a < xmax_b <= xmax_a and (ymin_a < ymax_b <= ymax_a or ymin_a <= ymin_b < ymax_a):
        if_intersect = True
    elif xmin_a <= xmin_b < xmax_a and (ymin_a < ymax_b <= ymax_a or ymin_a <= ymin_b < ymax_a):
        if_intersect = True
    elif xmin_b < xmax_a <= xmax_b and (ymin_b < ymax_a <= ymax_b or ymin_b <= ymin_a < ymax_b):
        if_intersect = True
    elif xmin_b <= xmin_a < xmax_b and (ymin_b < ymax_a <= ymax_b or ymin_b <= ymin_a < ymax_b):
        if_intersect = True
    else:
        return if_intersect
    if if_intersect:
        x_sorted_list = sorted([xmin_a, xmax_a, xmin_b, xmax_b])
        y_sorted_list = sorted([ymin_a, ymax_a, ymin_b, ymax_b])
        x_intersect_w = x_sorted_list[2] - x_sorted_list[1]
        y_intersect_h = y_sorted_list[2] - y_sorted_list[1]
        area_inter = x_intersect_w * y_intersect_h
        return area_inter


# IOU Part 2
def calcIOU(ver1, vertice2):
    # vertices in four points
    vertice1 = [ver1[0], ver1[1], ver1[0] + ver1[2], ver1[1] + ver1[3]]
    area_inter = if_intersection(vertice1[0], vertice1[2], vertice1[1], vertice1[3], vertice2[0], vertice2[2],
                                 vertice2[1], vertice2[3])
    if area_inter:
        area_1 = ver1[2] * ver1[3]
        area_2 = vertice2[4] * vertice2[5]
        iou = float(area_inter) / (area_1 + area_2 - area_inter)
        return iou

    # no intersection exist between this two squares
    return False

def doTrainingOnBBoxRegression():
    # load data
    # generateDataForRegression("./regression_train/2.txt", "./regression_train/2")
    # if training data was not prepared, it will be generated in fitting function

    net = RCNN_output.create_alexnet()
    model = tflearn.DNN(net)
    model.load("./fine_tune_model/fine_tune_model_save.model")

    regressionModelDict = {}

    # load bbox_regression_models been finished
    regressionDir = "./regression_train"
    for file in os.listdir(regressionDir):
        if file.split('_')[-1] == 'regression.pkl':
            regressionModel = joblib.load(os.path.join(regressionDir, file))
            objectClassTag = file.split("_")[0]
            # regressionModelArr.append(regressionModel)
            regressionModelDict[objectClassTag] = regressionModel

            # ps : about joblib (save and use model_data after finish its training)
            # >>> joblib.dump(clf, 'filename.pkl')  (save model_data)
            # restore
            # >>> clf = joblib.load('filename.pkl') (restore model_data)

    # if not exist, train bbox regression models
    if len(regressionModelDict) == 0:
        print("regression models not exist, training new ones.")
        train_file_folder = './regression_train'
        # model = alex_net  ( ??? 1.using alex_net to realise the function of preprocess
        # ( 2. and then, feed data out from preprocess to svmLinearSVC)
        regressionModelDict = fittingBBoxRegression(train_file_folder, model)

    print(" ------ finish fitting bbox_regressions.")
    return regressionModelDict

if __name__ == "__main__":
    # # trainBBoxRegression()
    # rectOfProposal = [2, 3, 234, 567]
    # rectOfNotation = [1, 4, 324, 666]
    # coef = calcCorrectionCoef(rectOfProposal, rectOfNotation)
    # print(coef)

    doTrainingOnBBoxRegression()
















    # def test_ridge_regression():
    #
    #
    #     # classifier = LogisticRegression(class_weight='balanced', solver='lbfgs', multi_class='multinomial', verbose=1,
    #     #                                 n_jobs=-1, max_iter=1000)
    #     # classifier.fit(X, y)
    #
    #     n_samples, n_features = 10, 5
    #     np.random.seed(0)
    #
    #     y = np.random.randn(n_samples)
    #     X = np.random.randn(n_samples, n_features)
    #     print("X :", X)
    #     print("y :", y)
    #     clf = Ridge(alpha=1.0)
    #     print(clf)
    #     # default parameters in Ridge()
    #     # Ridge(alpha=1.0, c
    #     # opy_X=True, fit_intercept=True, max_iter=None,
    #     #       normalize=False, solver='auto', tol=0.001)
    #     clf.fit(X, y)
    #     testX = np.array([0.14404357,1.45427351,0.76103773,0.12167502,0.44386323])
    #     pred_y = clf.predict([testX])
    #     print("pred_y :", pred_y)
