"""
可视化地验证算法各部分函数运行情况
"""

import cv2
from config import *
from constants import TrafficLightColor
from hsv_process import apply_mask, get_avg_saturation
from load_data import load_dataset
from matplotlib import pyplot as plt


def viz_load_data(image_list, red_index, yellow_index, green_index) -> None:
    """
    数据加载部分的可视化
    """
    _, ax = plt.subplots(1, 3, figsize=(5, 2))

    # red
    red_img = image_list[red_index][0]
    ax[0].imshow(red_img)
    ax[0].annotate(image_list[red_index][1].name, xy=(2, 5), color="red", fontsize="10")
    ax[0].axis("off")
    ax[0].set_title(red_img.shape, fontsize=10)

    # yellow
    yellow_img = image_list[yellow_index][0]
    plt.imshow(yellow_img)
    ax[1].imshow(yellow_img)
    ax[1].annotate(
        image_list[yellow_index][1].name, xy=(2, 5), color="yellow", fontsize="10"
    )
    ax[1].axis("off")
    ax[1].set_title(yellow_img.shape, fontsize=10)

    # green
    green_img = image_list[green_index][0]
    plt.imshow(green_img)
    ax[2].imshow(green_img)
    ax[2].annotate(
        image_list[green_index][1].name, xy=(2, 5), color="green", fontsize="10"
    )
    ax[2].axis("off")
    ax[2].set_title(green_img.shape, fontsize=10)
    plt.show()


def viz_hsv(image_list, image_num: int = 0) -> None:
    """
    将图像分解到hsv三通道的可视化
    """

    test_im, test_label = image_list[image_num]
    # convert to hsv
    hsv = cv2.cvtColor(test_im, cv2.COLOR_RGB2HSV)
    # Print image label
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    # Plot the original image and the three channels
    _, ax = plt.subplots(1, 4, figsize=(20, 10))
    ax[0].set_title("Standardized image")
    ax[0].imshow(test_im)
    ax[1].set_title("H channel")
    ax[1].imshow(h, cmap="gray")
    ax[2].set_title("S channel")
    ax[2].imshow(s, cmap="gray")
    ax[3].set_title("V channel")
    ax[3].imshow(v, cmap="gray")
    plt.show()


def viz_mask(rgb_image) -> None:
    """
    主算法中掩码作用后的可视化
    """

    hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)

    avg_saturation = get_avg_saturation(rgb_image)
    sat_low = int(avg_saturation * SATURATION_LOWER_RATIO)
    val_low = VALUE_LOWER

    red_result = apply_mask(rgb_image, sat_low, val_low, RED_LOWER, RED_UPPER)
    yellow_result = apply_mask(rgb_image, sat_low, val_low, YELLOW_LOWER, YELLOW_UPPER)
    green_result = apply_mask(rgb_image, sat_low, val_low, GREEN_LOWER, GREEN_UPPER)

    _, ax = plt.subplots(1, 4, figsize=(20, 10))
    ax[0].set_title("rgb image")
    ax[0].imshow(rgb_image)
    ax[1].set_title("red result")
    ax[1].imshow(red_result)
    ax[2].set_title("yellow result")
    ax[2].imshow(yellow_result)
    ax[3].set_title("green result")
    ax[3].imshow(green_result)
    plt.show()


if __name__ == "__main__":
    IMAGE_LIST = load_dataset(IMAGE_DIR_TRAINING)
    img_red = IMAGE_LIST[7][0]
    img_yellow = IMAGE_LIST[730][0]  # TODO 处理起始索引，根据实际情况变化
    img_green = IMAGE_LIST[800][0]
    img_test = [
        (img_red, TrafficLightColor.RED),
        (img_yellow, TrafficLightColor.YELLOW),
        (img_green, TrafficLightColor.GREEN),
    ]

    viz_load_data(IMAGE_LIST, 7, 730, 800)
    viz_hsv(IMAGE_LIST, 7)
    for img in img_test:
        viz_mask(img[0])
