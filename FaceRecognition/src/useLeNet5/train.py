import os

import matplotlib.image as mpimg
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from FaceRecognition.src.useLeNet5.model import defineModel

from FaceRecognition.src.useLeNet5.readData import load_data



def trainModel(dataset, model_dir, model_path):
    # train_set_x = data[0][0]
    # train_set_y = data[0][1]
    # valid_set_x = data[1][0]
    # valid_set_y = data[1][1]
    # test_set_x = data[2][0]
    # test_set_y = data[2][1]
    # X = tf.placeholder(tf.float32, shape=(None, None), name="x-input")  # 输入数据
    # Y = tf.placeholder(tf.float32, shape=(None, None), name='y-input')  # 输入标签

    batch_size = 40

    # train_set_x, train_set_y = dataset[0]
    # valid_set_x, valid_set_y = dataset[1]
    # test_set_x, test_set_y = dataset[2]
    train_set_x = dataset[0][0]
    train_set_y = dataset[0][1]

    valid_set_x = dataset[1][0]
    valid_set_y = dataset[1][1]

    test_set_x = dataset[2][0]
    test_set_y = dataset[2][1]

    X = tf.placeholder(tf.float32, [batch_size, 57 * 47])
    Y = tf.placeholder(tf.float32, [batch_size, 40])

    predict = defineModel(X)
    cost_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=Y))
    optimizer = tf.train.AdamOptimizer(1e-2).minimize(cost_func)

    # 用于保存训练的最佳模型
    saver = tf.train.Saver()
    # model_dir = './model'
    # model_path = model_dir + '/best.ckpt'
    with tf.Session() as session:
        # define training times
        trainingTimes = 50
        # 若不存在模型数据，需要训练模型参数
        if not os.path.exists(model_path + ".index"):
            print("No model can be used, start training for %s times ......" % str(trainingTimes))
            session.run(tf.global_variables_initializer())
        else:
            trainingTimes = 10
            print("Model exists, start training for only %s times ......" % str(trainingTimes))
            saver.restore(session, model_path)

        # start training
        best_loss = float('Inf')
        for epoch in range(trainingTimes):
            epoch_loss = 0
            for i in range((int)(np.shape(train_set_x)[0] / batch_size)):
                x = train_set_x[i * batch_size: (i + 1) * batch_size]
                y = train_set_y[i * batch_size: (i + 1) * batch_size]
                _, cost = session.run([optimizer, cost_func], feed_dict={X: x, Y: y})
                epoch_loss += cost

            print(epoch, ' : ', epoch_loss)
            if epoch_loss <= best_loss or abs(best_loss - epoch_loss) <= 1e-1:
                best_loss = epoch_loss
                if not os.path.exists(model_dir):
                    os.mkdir(model_dir)
                    print("create the directory: %s" % model_dir)
                save_path = saver.save(session, model_path)
                print("Model saved in file: %s" % save_path)

        # 恢复数据并校验和测试
        saver.restore(session, model_path)
        correct = tf.equal(tf.argmax(predict, 1), tf.argmax(Y, 1))
        valid_accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('valid set accuracy: ', valid_accuracy.eval({X: valid_set_x, Y: valid_set_y}))

        test_pred = tf.argmax(predict, 1).eval({X: test_set_x})
        test_true = np.argmax(test_set_y, 1)
        test_correct = correct.eval({X: test_set_x, Y: test_set_y})
        incorrect_index = [i for i in range(np.shape(test_correct)[0]) if not test_correct[i]]
        for i in incorrect_index:
            print('picture person is %i, but mis-predicted as person %i'
                  % (test_true[i], test_pred[i]))
        plot_errordata(incorrect_index, "../../data/olivettifaces.gif")


# 画出在测试集中错误的数据
def plot_errordata(error_index, dataset_path):
    img = mpimg.imread(dataset_path)
    plt.imshow(img)
    currentAxis = plt.gca()
    for index in error_index:
        row = index // 2
        column = index % 2
        currentAxis.add_patch(
            patches.Rectangle(
                xy=(47 * 9 if column == 0
                    else 47 * 19,
                    row * 57
                    ),
                width=47,
                height=57,
                linewidth=1,
                edgecolor='r',
                facecolor='none'
            )
        )
    plt.savefig("result.png")
    plt.show()


def run():
    dataset_path = "../../data/olivettifaces.gif"
    data = load_data(dataset_path)
    model_dir = './model'
    model_path = model_dir + '/best.ckpt'
    trainModel(data, model_dir, model_path)


if __name__ == "__main__":
    run()