"""
从磁盘中加载图像文件，转为矩阵数据、添加标签
"""

import glob
import os.path

import numpy as np
from constants import TrafficLightColor
from matplotlib import image as mpimg
from PIL import UnidentifiedImageError


def load_dataset(image_dir: str) -> list[tuple[np.ndarray, TrafficLightColor]]:
    """
    加载图像数据与添加标签
    :param image_dir: 图像目录
    :return: image 与 label
    """

    image_list = []

    # 遍历每个颜色文件夹
    for image_type in TrafficLightColor:
        file_lists = glob.glob(os.path.join(image_dir, image_type.value, "*"))
        for file in file_lists:
            try:
                image = mpimg.imread(file)
                image_list.append((image, image_type))
            except UnidentifiedImageError:
                # 若某文件无法作为图像被读取加载，则跳过
                continue

    return image_list
