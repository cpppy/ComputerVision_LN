from __future__ import division, print_function, absolute_import
import os.path
import preprocessing_RCNN
import config
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression



# Use a already trained alexnet with the last layer redesigned
def create_alexnet(num_classes, restore=False):
    # Building 'AlexNet'
    network = input_data(shape=[None, 224, 224, 3])
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
    network = dropout(network, 0.5)
    # do not restore this layer
    network = fully_connected(network, num_classes, activation='softmax', restore=restore)
    network = regression(network, optimizer='momentum',
                         loss='categorical_crossentropy',
                         learning_rate=0.001)
    return network


def fineTuneTrainAlexnet(network,
                         childImageArr,
                         childImageLabelArr,
                         preTrainModelPath,
                          fineTuneTrainModelPath):
    # Training
    model = tflearn.DNN(network, checkpoint_path='rcnn_model_alexnet',
                        max_checkpoints=1, tensorboard_verbose=2, tensorboard_dir='output_RCNN')
    if os.path.isfile(fineTuneTrainModelPath):
        print("Loading the fine tuned model.")
        model.load(fineTuneTrainModelPath)

    elif os.path.isfile(preTrainModelPath):
        print("Loading the alexnet")
        model.load(preTrainModelPath)
    else:
        print("No file to load, error")
        return False

    model.fit(childImageArr, childImageLabelArr, n_epoch=1, validation_set=0.1, shuffle=True,
              show_metric=True, batch_size=32, snapshot_step=100,
              snapshot_epoch=False, run_id='alexnet_rcnnflowers2')
    # Save the model
    model.save(fineTuneTrainModelPath)
    print("save data for fine_tune_train_model success.")










if __name__ == '__main__':

    npyDirPath = "./data_set"
    # firstly, check npy_files exist or not ?
    #   if exists, continue
    #   else  generate it !
    if len(os.listdir(npyDirPath)) == 0:
        print("...no data get from path : " + npyDirPath)
        print("Reading Data, function = load_train_proposal, dataSource: " + "./fine_tune_list.txt")
        preprocessing_RCNN.loadDataFromFile("./fine_tune_list.txt",
                                         npyDirPath,
                                         2)
    print("Loading Data, function = load_from_npy")
    childImageArr, childImageLabelArr = preprocessing_RCNN.loadDataFromNpyFile(npyDirPath)
    restore = False
    if os.path.isfile("./fine_tune_model/fine_tune_model_save.model.index"):
        restore = True
        print("...found model variable data for fine_tune_train_model exists in directory, continue fine-tune !")

        # if fine_tune_training has been executed,
        #    the variable_data of last fc_layer will be restored by training result
        # else   (use data of pre_traning(train_alexnet)
        #    just restore vars of other layers by training result of (training_alexnet process),
        #                                                           excluding the last fc_layer
        #    the last  fc_layer will be initialization by truncated random value

    # three classes(include background)
    # num_class = 3, because there are 2 kinds of flowers in data,
    # but, when use ss&proposalRegions, it will generate one more as background
    net = create_alexnet(3, restore=restore)
    fineTuneTrainAlexnet(net,
                         childImageArr,
                         childImageLabelArr,
                         "./pre_train_model/model_save.model.index",
                         "./fine_tune_train_model/fine_tune_model_save.model.index")
