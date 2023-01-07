"""
算法主函数、完整运行算法入口
"""

import time

import numpy as np
from config import *
from constants import TrafficLightColor
from hsv_process import apply_mask, find_none_zero, get_avg_saturation
from load_data import load_dataset
from standardize import standardize_image

# 加载数据
IMAGE_LIST = load_dataset(IMAGE_DIR_TRAINING)

# 标准化
standardized_train_list = standardize_image(IMAGE_LIST, *STD_IMAGE_SIZE)


def traffic_light_classification(rgb_image: np.ndarray) -> TrafficLightColor:
    """
    算法主函数，返回输入的图像对应的信号灯颜色 \n
    算法参数（变量名全大写）需要在配置文件中设置 \n
    :param rgb_image: rgb空间下的单张图像数据
    :return: 信号灯颜色（枚举值）
    """

    avg_saturation = get_avg_saturation(rgb_image)  # 平均饱和度
    sat_low = int(avg_saturation * SATURATION_LOWER_RATIO)
    val_low = VALUE_LOWER

    red_result = apply_mask(rgb_image, sat_low, val_low, RED_LOWER, RED_UPPER)
    yellow_result = apply_mask(rgb_image, sat_low, val_low, YELLOW_LOWER, YELLOW_UPPER)
    green_result = apply_mask(rgb_image, sat_low, val_low, GREEN_LOWER, GREEN_UPPER)

    sum_red = find_none_zero(red_result)
    sum_yellow = find_none_zero(yellow_result)
    sum_green = find_none_zero(green_result)
    sum_max = max(sum_red, sum_yellow, sum_green)

    if sum_max == 0:
        return TrafficLightColor.UNIDENTIFIED
    elif sum_max == sum_red:
        return TrafficLightColor.RED
    elif sum_max == sum_yellow:
        return TrafficLightColor.YELLOW
    elif sum_max == sum_green:
        return TrafficLightColor.GREEN
    else:
        return TrafficLightColor.UNIDENTIFIED


# Constructs a list of misclassified images given a list of test images and their labels
# This will throw an assertion error if labels are not standardized(one hot encode)
def get_misclassified_images(test_images):
    misclassified_images_labels = []
    # Iterate through all the test images
    # Classify each image  and compare to the true label
    for image in test_images:
        # Get true data
        im = image[0]
        true_label = image[1]
        assert (
            type(true_label) == TrafficLightColor
        ), "This true_label is not the excepted type."

        # Get predicted label from your classifier
        predicted_label = traffic_light_classification(im)
        assert (
            type(predicted_label) == TrafficLightColor
        ), "This predicted_label is not the excepted type."

        # compare true and predicted labels
        if predicted_label != true_label:
            # if these labels are ot equal, the image  has been misclassified
            misclassified_images_labels.append((im, predicted_label, true_label))

    return misclassified_images_labels


if __name__ == "__main__":
    start = time.process_time_ns()
    # Find all misclassified images in a given test set
    MISCLASSIFIED = get_misclassified_images(standardized_train_list)
    end = time.process_time_ns()
    print("time = ", end - start)

    # 准确度计算
    total = len(standardized_train_list)
    num_correct = total - len(MISCLASSIFIED)
    accuracy = num_correct / total
    print("Accuracy:" + str(accuracy))
    print(
        "Number of misclassified images = "
        + str(len(MISCLASSIFIED))
        + " out of "
        + str(total)
    )
