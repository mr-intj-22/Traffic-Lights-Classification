"""
算法配置文件
"""

# 数据集文件路径
IMAGE_DIR_TRAINING: str = "../traffic_light_images/training"
IMAGE_DIR_TEST: str = "../traffic_light_images/test"

# 标准化图像尺寸
STD_IMAGE_SIZE: tuple[int, int] = (32, 64)  # (width, height)

# HSV阈值控制
SATURATION_LOWER_RATIO: float = 1.3  # 平均饱和度乘以此系数，作为饱和度下限，推荐值1.3
VALUE_LOWER: int = 140  # 亮度下限
RED_LOWER: int = 150  # 红色色相下限
RED_UPPER: int = 180  # 红色色相上限
YELLOW_LOWER: int = 10  # 黄色色相下限
YELLOW_UPPER: int = 60  # 黄色色相上限
GREEN_LOWER: int = 70  # 绿色色相下限
GREEN_UPPER: int = 100  # 绿色色相上限
