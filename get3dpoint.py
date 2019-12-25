import ctypes
import os
import numpy as np

so1 = ctypes.cdll.LoadLibrary(os.getcwd() + '/libget3dfixeds1.so')


def get3dpointwuliu(Left_DetectPoints, Right_DetectPoints):
    """

    :param Left_DetectPoints: 左眼的检测点
    :param Right_DetectPoints: 右眼的监测点
    :return: 采摘点
    """
    so1.LOADcameraparam()
    c_double_p = ctypes.POINTER(ctypes.c_double)
    so1.Get3DPoint.argtypes = (ctypes.POINTER(ctypes.c_double),  ctypes.POINTER(ctypes.c_double))
    so1.Get3DPoint.restype = ctypes.c_int  # 函数 Get3DPoint 返回值类型

    Left_DetectPoints = np.asarray(Left_DetectPoints, dtype=np.float64)
    print(Left_DetectPoints)

    args1, args2 = Left_DetectPoints.shape  # Left_DetectPoints的维度
    Left_DetectPoints_LenP = args1 * args2
    Left_DetectPoints_Len = ctypes.c_int(Left_DetectPoints_LenP)

    # 判断是否为连续内存,如果不是则转换为连续内存存储
    if not Left_DetectPoints.flags['C_CONTIGUOUS']:
        Left_DetectPoints = np.ascontiguousarray(Left_DetectPoints, dtype=Left_DetectPoints.dtype)
    Left_DetectPoints = Left_DetectPoints.ctypes.data_as(c_double_p)  # 转换为指针类型

    # Left_DetectPoints = Left_DetectPoints.ctypes.data_as(ctypes.c_char_p)

    Right_DetectPoints = np.asarray(Right_DetectPoints, dtype=np.float64)  # np.mauchart('1 2 3; 6 6 4; 0.3 2.9 3.7')
    args3, args4 = Right_DetectPoints.shape
    Right_DetectPoints_LenP = args3 * args4
    Right_DetectPoints_Len = ctypes.c_int(Right_DetectPoints_LenP)

    if not Right_DetectPoints.flags['C_CONTIGUOUS']:
        Right_DetectPoints = np.ascontiguousarray(Right_DetectPoints, dtype=Right_DetectPoints.dtype)
    Right_DetectPoints = Right_DetectPoints.ctypes.data_as(c_double_p)  # 转换为指针类型

    result_p = np.random.randint(2, size=((Right_DetectPoints_LenP if (Right_DetectPoints_LenP > Left_DetectPoints_LenP) else Left_DetectPoints_LenP)*3)).ctypes.data_as(c_double_p)

    result_len = so1.Get3DPoint(Left_DetectPoints, Right_DetectPoints, Left_DetectPoints_Len, Right_DetectPoints_Len, result_p)
    result = np.fromiter(result_p, dtype=np.float64, count=result_len).reshape(-1, 3)
    return result




