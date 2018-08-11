from __future__ import division, print_function, absolute_import

import math
import os
import os.path

import cv2
import numpy as np
import tflearn
from sklearn import svm
from sklearn.externals import joblib
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import local_response_normalization

import boundingbox_regression
import config
import preprocessing_RCNN as prep
import selectivesearch
import tools
import NMS_filter


def image_proposal(img_path):
    img = cv2.imread(img_path)
    img_lbl, regions = selectivesearch.selective_search(
                       img, scale=500, sigma=0.9, min_size=10)
    candidates = set()
    images = []
    vertices = []
    for r in regions:
        # excluding same rectangle (with different segments)
        if r['rect'] in candidates:
            continue
        # excluding small regions
        if r['size'] < 220:
            continue
        if (r['rect'][2] * r['rect'][3]) < 500:
            continue
        # resize to 227 * 227 for input
        proposal_img, proposal_vertice = prep.cropImage(img, r['rect'])
        # Delete Empty array
        if len(proposal_img) == 0:
            continue
        # Ignore things contain 0 or not C contiguous array
        x, y, w, h = r['rect']
        if w == 0 or h == 0:
            continue
        # Check if any 0-dimension exist
        [a, b, c] = np.shape(proposal_img)
        if a == 0 or b == 0 or c == 0:
            continue
        resized_proposal_img = prep.resize_image(proposal_img, config.IMAGE_SIZE, config.IMAGE_SIZE)
        candidates.add(r['rect'])
        img_float = np.asarray(resized_proposal_img, dtype="float32")
        images.append(img_float)
        vertices.append(r['rect'])
    return images, vertices


# Load training images
# train_file : ./svm_train/1.txt   2.txt
def generate_single_svm_train(train_file):
    save_path = train_file.rsplit('.', 1)[0].strip()
    if len(os.listdir(save_path)) == 0:
        print("reading %s's svm dataset" % train_file.split('\\')[-1])

        # generate region proposals for images in 1.txt/ 2.txt,
        # and store it into a npy file
        prep.load_train_proposals(train_file,
                                  2,
                                  save_path,
                                  threshold=0.3,
                                  is_svm=True,
                                  save=True)
    print("restoring svm dataset")
    images, labels = prep.load_from_npy(save_path)

    return images, labels


# Use a already trained alexnet with the last layer redesigned
def create_alexnet():
    # Building 'AlexNet'
    network = input_data(shape=[None, config.IMAGE_SIZE, config.IMAGE_SIZE, 3])
    network = conv_2d(network, 96, 11, strides=4, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 256, 5, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 256, 3, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, 4096, activation='tanh')
    network = regression(network, optimizer='momentum',
                         loss='categorical_crossentropy',
                         learning_rate=0.001)
    return network


# Construct cascade svms
def train_svms(train_file_folder, model):
    files = os.listdir(train_file_folder) # './svm_train'
    svms = []
    for train_file in files:
        # get "1.txt" and "2.txt"
        # means two kinds of flowers
        # then, train two svm_models for this two kinds of flowers
        if train_file.split('.')[-1] == 'txt':
            filePath = os.path.join(train_file_folder, train_file)
            X, Y = generate_single_svm_train(filePath)
            train_features = []
            for index, image in enumerate(X):
                # extract features
                features = model.predict([image])

                # features output from alex_net, as the input of svm
                train_features.append(features[0])
                tools.view_bar("extract features of %s" % train_file, index + 1, len(X))
            print(' ')
            print("feature dimension")
            print(np.shape(train_features))
            # SVM training
            clf = svm.SVC(kernel="linear", probability=True)
            print("fit svm")
            clf.fit(train_features, Y)
            svms.append(clf)

            # use joblib package to store svm_training result
            joblib.dump(clf, os.path.join(train_file_folder, str(train_file.split('.')[0]) + '_svm.pkl'))

    return svms

def adjustByBBoxRegression(classifyLabelArr, rectArr, featureArr):
    print("adjust rectangle params by bbox regression model.")
    regressionModelDict = {}
    resLabelArr = []
    resRectArr = []
    # firstly, search bbox_regression_model_data in "./regression_train"
    # if exist, do regressionModelArr.append()
    for file in os.listdir("./regression_train"):
        print("current file :", file)
        if file.__contains__("_") and file.split('_')[-1] == 'regression.pkl':
            print(" +++ detect bbox_regression_model file, load it... ")
            objectClassifyTag = file.split("_")[0]
            print(" object classify tag :", objectClassifyTag)
            regModel = joblib.load(os.path.join("./regression_train", file))
            # regressionModelArr.append(regModel)
            regressionModelDict[objectClassifyTag] = regModel

    # if not exist, train svm models
    if len(regressionModelDict) == 0:
        print("regression models need to be generated.")
        regressionModelDict = boundingbox_regression.doTrainingOnBBoxRegression()
    print("Done fitting bbox regression model.")

    print("---> begin check")
    for index, label in enumerate(classifyLabelArr):
        print(" currIndex :", index)
        print(" currlabel :", int(label))
        if (regressionModelDict.__contains__(str(int(label)))):
            print(" +++ get matched regression model.")
            checkedModel = regressionModelDict[str(int(label))]
            predCoefResult = checkedModel.predict([featureArr[index]])
            correctionCoef = predCoefResult[0]
            print("rect arr before adjusting :", rectArr[index])
            print("correctionCoef :", correctionCoef)

            resRect = doCorrectionByCoef(rectArr[index], correctionCoef)
            # add result to output arr
            resLabelArr.append(label)
            resRectArr.append(resRect)
            print("rect arr after adjusting :", resRect)
    return resLabelArr, resRectArr


def doCorrectionByCoef(rectArr, correctionCoef):
    Px = rectArr[0]
    Py = rectArr[1]
    Pw = rectArr[2]
    Ph = rectArr[3]

    tx = correctionCoef[0]
    ty = correctionCoef[1]
    tw = correctionCoef[2]
    th = correctionCoef[3]

    Gx = Px + Pw*tx
    Gy = Py + Py*ty
    Gw = Pw*math.exp(tw)
    Gh = Ph*math.exp(th)

    rectArrAfterCorrect = np.array([Gx, Gy, Gw, Gh])
    resRectArr = np.array([int(v) for v in rectArrAfterCorrect])
    return resRectArr




if __name__ == '__main__':
    train_file_folder = config.TRAIN_SVM  # './svm_train'

    # (directory 7 and 16) was not used for training, so take it into testing
    img_path = './17flowers/jpg/7/image_0580.jpg'  # or './17flowers/jpg/16/****.jpg'

    # get region proposals by selective search
    # return is child_imgs(float) & vertices/rectangle(position info for this child_img)
    imgs, verts = image_proposal(img_path)

    #display this image and tagging the region box
    # tools.show_rect(img_path, verts)

    # just define the alexNet for RCNN_output
    net = create_alexnet()
    # generate a nn_package for this nn_model
    model = tflearn.DNN(net)

    # load model data gotten by training before
    # FINE_TUNE_MODEL_PATH = './fine_tune_train_model/fine_tune_model_save.model'
    model.load(config.FINE_TUNE_MODEL_PATH)

    # svm for classification
    svms = []

    # firstly, search svm_model_data in "./svm_train"
    # if exist, do svms.append()
    for file in os.listdir(train_file_folder):
        if file.split('_')[-1] == 'svm.pkl':
            # it means that, a svm classification model finished and stored in this file
            # what to do the next is : sppending this model to svms_arr
            svms.append(joblib.load(os.path.join(train_file_folder, file)))

            # ps : about joblib (save and use model_data after finish its training)
            # >>> from sklearn.externals import joblib
            # >>> clf = svm.SVC()
            # save
            # >>> joblib.dump(clf, 'filename.pkl')  (save model_data)
            # restore
            # >>> clf = joblib.load('filename.pkl') (restore model_data)

    # if not exist, train svm models
    if len(svms) == 0:
        # train_file_folder = './svm_train'
        # model = alex_net  ( ??? 1.using alex_net to realise the function of preprocess
        # ( 2. and then, feed data out from preprocess to svmLinearSVC)
        svms = train_svms(train_file_folder, model)

    print("Done fitting svms")

    # firstly, get feartures if images by alex_net
    features = model.predict(imgs)
    print("predict image:")
    print(np.shape(features))

    results_rect = []
    results_label = []
    results_proba = []
    results_feature = []
    for featureIndex, feature in enumerate(features):

        # init decision of current bounding box
        maxProba = -1
        labelInMaxProba = 0
        rectInMaxProba = []

        for svmIndex, svm in enumerate(svms):
            svmClassLabel = svmIndex + 1
            print("---> svm index :", svmClassLabel)
            pred = svm.predict_proba([feature.tolist()])
            probaArr = pred[0]
            # not background
            if (probaArr[0]<probaArr[1]):
                print(" +++ detect object in this childImg.")
                print("proba :", probaArr[1])
                if (probaArr[1] > maxProba):
                    print("larger probability appear in this svm model.")
                    maxProba = probaArr[1]
                    labelInMaxProba = svmClassLabel
                    rectInMaxProba = verts[featureIndex]
        # use predict result on max probability as the final result
        if (maxProba > 0):
            results_label.append(labelInMaxProba)
            results_rect.append(rectInMaxProba)
            results_proba.append(maxProba)
            results_feature.append(feature)

    print("result_rect:", results_rect)
    print("result_label:", results_label)
    print("result_proba:", results_proba)

    labelChoosed = 1
    results_label_1 = []
    results_rect_1 = []
    results_proba_1 = []
    results_feature_1 = []
    for index, label in enumerate(results_label):
        print(index, label)
        if (label == labelChoosed):
            results_label_1.append(label)
            results_rect_1.append(results_rect[index])
            results_proba_1.append(results_proba[index])
            results_feature_1.append(results_feature[index])

    print("resRectArr of label=1 :", results_rect_1)

    # NMS
    resRectArr_1, resProbaArr_1, checkedIndex_1 = NMS_filter.filterByNMS(results_rect_1, results_proba_1)
    resLabelArr_1 = np.ones(len(resRectArr_1))
    resFeatureArr_1 = []
    for index in checkedIndex_1:
        resFeatureArr_1.append(results_feature[index])

    # BBox Regression
    resLabelArr, resRectArr = adjustByBBoxRegression(resLabelArr_1,
                                                     resRectArr_1,
                                                     resFeatureArr_1)

    tools.show_rect(img_path, resRectArr)


#
# def outputOnImg(results):
#     img = skimage.io.imread(img_path)
#     fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
#     ax.imshow(img)
#     for x, y, w, h in results:
#         rect = mpatches.Rectangle(
#             (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
#         ax.add_patch(rect)
#
#         plt.show()







