"""
对原始图片输入进行标准化
"""

import cv2
import numpy as np
from constants import TrafficLightColor


def standardize_image(
    image_list: list[tuple[np.ndarray, TrafficLightColor]],
    width: int,
    height: int,
) -> list[tuple[np.ndarray, TrafficLightColor]]:
    """
    将带有标签的图像列表输入逐个进行标准化处理，返回处理后的图像列表
    :param image_list: 图像与标签列表
    :param width: 标准图像宽度
    :param height: 标准图像高度
    :return: 标准化的图像列表
    """

    standard_list = []

    for item in image_list:
        image, label = item
        standardized_im = standardize_input(image, width, height)  # 标准化输入
        standard_list.append((standardized_im, label))

    return standard_list


def standardize_input(image: np.ndarray, width: int, height: int):
    """
    辅助函数，将图像缩放到指定尺寸
    :param image: 原始图像
    :param width: 缩放后宽度
    :param height: 缩放后高度
    :return: 缩放后的图像
    """

    standard_image = cv2.resize(image, (width, height))
    return standard_image
