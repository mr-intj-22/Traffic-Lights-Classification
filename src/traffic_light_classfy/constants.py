"""
常量与枚举值
"""

import enum


@enum.unique
class TrafficLightColor(enum.Enum):
    """
    交通灯颜色类型标签
    注意其value为文件系统中目录名称
    """

    RED = "red"
    YELLOW = "yellow"
    GREEN = "green"
    UNIDENTIFIED = "unidentified"
