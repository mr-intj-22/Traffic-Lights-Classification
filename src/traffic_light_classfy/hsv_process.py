"""
HSV空间下的图像处理
"""

import cv2
import numpy as np


def get_avg_value(rgb_image: np.ndarray) -> float:
    """
    计算平均亮度
    :param rgb_image: rgb图像
    :return: 平均亮度
    """

    hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    v_channel = hsv_image[:, :, 2]
    return np.average(v_channel)


def get_avg_saturation(rgb_image: np.ndarray) -> float:
    """
    计算平均饱和度
    :param rgb_image: rgb图像
    :return: 平均饱和度
    """

    hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    s_channel = hsv_image[:, :, 1]
    return np.average(s_channel)


def apply_mask(
    rgb_image: np.ndarray, saturation_lower, value_lower, color_lower, color_upper
) -> np.ndarray:
    """
    对输入图像应用指定范围的掩膜，颜色由HSV中的上下边界确定
    :param rgb_image: 原始图像（RGB）
    :param saturation_lower: 饱和度下界
    :param value_lower: 亮度下界
    :param color_lower: 颜色下界（HSV）
    :param color_upper: 颜色上界（HSV）
    :return: 应用掩膜后的图像
    """

    hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    lower = np.array([color_lower, saturation_lower, value_lower])
    upper = np.array([color_upper, 255, 255])
    red_mask = cv2.inRange(hsv_image, lower, upper)
    image_result = cv2.bitwise_and(rgb_image, rgb_image, mask=red_mask)

    return image_result


def find_none_zero(rgb_image: np.ndarray) -> int:
    """辅助函数"""
    # TODO 通过numpy向量化重写此函数，提升性能
    rows, cols, _ = rgb_image.shape
    counter = 0
    for row in range(rows):
        for col in range(cols):
            pixels = rgb_image[row, col]
            if sum(pixels) != 0:
                counter = counter + 1

    return counter
