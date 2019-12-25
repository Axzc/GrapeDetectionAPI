import base64
import cv2
from io import BytesIO

import numpy as np
from PIL import Image


def verify_param(abort, logger, **kwargs):
    """
    验证参数是否为空
    :param abort: 异常处理
    :param logger: 日志
    :param kwargs: 要验证的参数
    :return: True
    """
    for key in kwargs:
        if kwargs[key] is None or kwargs[key] == '':
            logger.error("{} param not right from method {}".format(key, kwargs["method_name"], exc_info=True))
            return abort(kwargs['error_code'], key)
    return True


def convert_to_cv2(abort, logger, image):
    """
    将图片转换为cv2
    :param abort: 异常处理
    :param logger: 日志
    :param image: 图片
    :return: cv2 格式的图片
    """

    starter = image.find(',')
    image_base64 = image[starter + 1:]
    try:
        bytes_buffer = BytesIO(base64.b64decode(image_base64))
        image = Image.open(bytes_buffer, mode='r')
    except:
        logger.error("Conversion to cv2 failed, param not right")
        abort(400)
    # fileNPArray = np.fromstring(bytes_buffer.getvalue(), np.uint8)    u
    # cv2_img = cv2.imdecode(fileNPArray, cv2.IMREAD_COLOR)
    cv2_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    return cv2_img