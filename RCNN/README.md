python + tensorflow + sklearn

1、opencv读取图片数据以及对应分类，作为数据训练alexnet，此为pre train

2、使用selectivesearch，生成数据，训练 fine_tune_train_model

3、生成数据训练svm，ss之后的子图选取作为样本时，IOU的阈值与2中不同

4、生成训练数据用于边框回归，bbox regression

5、对图片进行物体检测时使用NMS筛选候选框