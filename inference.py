from __future__ import division
import cv2
import sys
import numpy as np
import tensorflow as tf
from keras_frcnn import roi_helpers
from keras_frcnn import resnet as nn
from keras import backend as K
from keras.layers import Input
from keras.models import Model

sys.setrecursionlimit(40000)


class Inference(object):

    loaded_model_rpn = None
    loaded_model_classifier = None
    graph = None
    session = None
    num_rois = 32
    # 分类类别
    class_mapping = {0: 'grape', 1: 'bg'}

    def __init__(self, model_path='./model_frcnn.hdf5', num_features=1024,):

        if Inference.loaded_model_classifier is not None and Inference.loaded_model_rpn is not None:
            return
        self.model_path = model_path
        # self.num_rois = num_rois
        self.num_features = num_features
        self.input_shape_img, self.input_shape_features = self.__set_img_format()
        Inference.session = tf.Session()
        Inference.graph = tf.get_default_graph()
        with Inference.graph.as_default():
            with Inference.session.as_default():
                # self.create_model()
                # self.model_rpn, self.model_classifier = self.load_model()
                self.load_model()

    def __set_img_format(self, ):
        num_features = self.num_features
        if K.image_dim_ordering() == 'th':
            input_shape_img = (3, None, None)
            input_shape_features = (num_features, None, None)
        else:
            input_shape_img = (None, None, 3)
            input_shape_features = (None, None, num_features)
        return input_shape_img, input_shape_features

    # 将边界框的坐标转换为其原始大小
    def __get_real_coordinates(self, ratio, x1, y1, x2, y2):
        real_x1 = int(round(x1 // ratio))
        real_y1 = int(round(y1 // ratio))
        real_x2 = int(round(x2 // ratio))
        real_y2 = int(round(y2 // ratio))

        return real_x1, real_y1, real_x2, real_y2

    # 格式化图片大小
    def __format_img_size(self, img, ):
        im_size = 600

        img_min_side = float(im_size)
        (height, width, _) = img.shape

        if width <= height:
            ratio = img_min_side / width
            new_height = int(ratio * height)
            new_width = int(img_min_side)
        else:
            ratio = img_min_side / height
            new_width = int(ratio * width)
            new_height = int(img_min_side)
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        return img, ratio

    # 根据配置格式化图像通道
    def __format_img_channels(self, img):

        img_channel_mean = [103.939, 116.779, 123.68]
        img_scaling_factor = 1.0
        
        img = img[:, :, (2, 1, 0)]
        img = img.astype(np.float32)
        img[:, :, 0] -= img_channel_mean[0]
        img[:, :, 1] -= img_channel_mean[1]
        img[:, :, 2] -= img_channel_mean[2]
        img /= img_scaling_factor
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        
        return img

    # 基于配置格式化图像以进行模型预测
    def __format_img(self, img):
        img, ratio = self.__format_img_size(img)
        img = self.__format_img_channels(img)
        return img, ratio        

    def __create_model(self):
        print('creating model')
        
        # 锚点的尺寸和比例
        anchor_box_ratios = [[1, 1], [1, 2], [2, 1]]
        anchor_box_scales = [128, 256, 512]

        img_input = Input(shape=self.input_shape_img)
        roi_input = Input(shape=(self.num_rois, 4))
        feature_map_input = Input(shape=self.input_shape_features)

        shared_layers = nn.nn_base(img_input, trainable=True)

        num_anchors = len(anchor_box_scales) * len(anchor_box_ratios)
        rpn_layers = nn.rpn(shared_layers, num_anchors)
        classifier = nn.classifier(feature_map_input, roi_input, self.num_rois, nb_classes=len(self.class_mapping),
                                   trainable=True)
        model_rpn = Model(img_input, rpn_layers)
        model_classifier = Model([feature_map_input, roi_input], classifier)

        Inference.loaded_model_rpn, Inference.loaded_model_classifier = model_rpn, model_classifier
        return

    def load_model(self):
        print('loading model')
        # 加载模型
        print('Loading weights from {}'.format(self.model_path))
        model_path = self.model_path
        self.__create_model()
        Inference.loaded_model_rpn.load_weights(model_path, by_name=True)
        Inference.loaded_model_classifier.load_weights(model_path, by_name=True)
        Inference.loaded_model_rpn.compile(optimizer='sgd', loss='mse')
        Inference.loaded_model_classifier.compile(optimizer='sgd', loss='mse')

        return True

    def pre_predict(self, img=None):
        print('judging object')
        with Inference.graph.as_default():
            with Inference.session.as_default():
                bboxes = {}
                probs = {}

                rpn_stride = 16
                bbox_threshold = 0.8
                classifier_regr_std = [8.0, 8.0, 4.0, 4.0]
                model_rpn = Inference.loaded_model_rpn
                model_classifier = Inference.loaded_model_classifier

                # 定义预测框
                # img = cv2.imread(pic_path)
                X, ratio = self.__format_img(img)

                if K.image_dim_ordering() == 'tf':
                    X = np.transpose(X, (0, 2, 3, 1))

                # 从RPN获取特征匹配和输出
                [Y1, Y2, F] = model_rpn.predict(X)

                # infer roi
                R = roi_helpers.rpn_to_roi(Y1, Y2, K.image_dim_ordering(), overlap_thresh=0.7)

                # 将(x1,y1,x2,y2)转化为(x,y,w,h)
                R[:, 2] -= R[:, 0]
                R[:, 3] -= R[:, 1]

                # 声明变量
                bboxes = {}
                probs = {}

                for jk in range(R.shape[0] // self.num_rois + 1):
                    ROIs = np.expand_dims(R[self.num_rois * jk:self.num_rois * (jk + 1), :], axis=0)
                    if ROIs.shape[1] == 0:
                        break
                    if jk == R.shape[0] // self.num_rois:
                        # pad R
                        curr_shape = ROIs.shape
                        target_shape = (curr_shape[0], self.num_rois, curr_shape[2])
                        ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
                        ROIs_padded[:, :curr_shape[1], :] = ROIs
                        ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
                        ROIs = ROIs_padded
                    [P_cls, P_regr] = model_classifier.predict([F, ROIs])
                    # 对每一个box筛选出预测结果中概率最高的物体类
                    for ii in range(P_cls.shape[1]):

                        if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                            continue

                        cls_name = self.class_mapping[np.argmax(P_cls[0, ii, :])]

                        if cls_name not in bboxes:
                            bboxes[cls_name] = []
                            probs[cls_name] = []

                        (x, y, w, h) = ROIs[0, ii, :]

                        cls_num = np.argmax(P_cls[0, ii, :])
                        try:
                            (tx, ty, tw, th) = P_regr[0, ii, 4 * cls_num:4 * (cls_num + 1)]
                            tx /= classifier_regr_std[0]
                            ty /= classifier_regr_std[1]
                            tw /= classifier_regr_std[2]
                            th /= classifier_regr_std[3]
                            x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
                        except:
                            pass
                        bboxes[cls_name].append(
                            [rpn_stride * x, rpn_stride * y, rpn_stride * (x + w), rpn_stride * (y + h)])
                        probs[cls_name].append(np.max(P_cls[0, ii, :]))

                return bboxes, probs, ratio

    def post_predict(self, img=None):
        print('confirming location')
        # 求出预测框的比例前原始坐标和真实坐标
        # origin_boxes = []
        real_boxes = []
        bboxes, probs, ratio = self.pre_predict(img=img)

        for key in bboxes:
            bbox = np.array(bboxes[key])
            # 非极大值抑制法求出待测物体坐标
            new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.7)
            for jk in range(new_boxes.shape[0]):
                # 预测框的比例前原始坐标
                (x1, y1, x2, y2) = new_boxes[jk, :]
                # 获得原图的真实坐标
                (real_x1, real_y1, real_x2, real_y2) = self.__get_real_coordinates(ratio, x1, y1, x2, y2)
                # 写入数据
                # origin_boxes.append([x1, y1, x2, y2])
                real_boxes.append([real_x1, real_y1, real_x2, real_y2])
        # 返回数据
        return real_boxes