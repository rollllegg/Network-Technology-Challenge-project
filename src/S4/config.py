MODE = "soft" # 模式选择，“soft”为mode b，其他是mode a, mode a为开发半成品，不推荐使用

csv_dir = "../S3/output/links" # csv文件夹路径，存储网络拓扑信息
rules_dir = "../S3/output/rules" # json文件夹路径，存储路由规则

from enum import Enum
class action(Enum):
    NOP = 1
    ADD = 2
    DEL = 3
    REPLACE = 4


import logging
import os
class LogColor:
    """
    根据不同的日志级别，打印不同颜色的日志，并将日志写入不同的文件
    info：绿色
    warning：黄色
    error：红色
    debug：灰色
    """
    # 确保日志文件夹存在
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    # 定义不同日志级别的文件
    info_log_file = os.path.join(log_dir, "info_runtime.log")
    warning_log_file = os.path.join(log_dir, "warning_runtime.log")
    error_log_file = os.path.join(log_dir, "error_runtime.log")
    debug_log_file = os.path.join(log_dir, "debug_runtime.log")

    # 配置不同的日志记录器
    info_logger = logging.getLogger("info_logger")
    warning_logger = logging.getLogger("warning_logger")
    error_logger = logging.getLogger("error_logger")
    debug_logger = logging.getLogger("debug_logger")

    for logger, log_file in [
        (info_logger, info_log_file),
        (warning_logger, warning_log_file),
        (error_logger, error_log_file),
        (debug_logger, debug_log_file)
    ]:
        handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s: %(message)s'))
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

    @staticmethod
    def info(message: str):
        # info级别的日志，绿色
        LogColor.info_logger.info(message)
        print("\033[0;32m" + message + "\033[0m")

    @staticmethod
    def warning(message: str):
        # warning级别的日志，黄色
        LogColor.warning_logger.warning(message)
        print("\033[0;33m" + message + "\033[0m")

    @staticmethod
    def error(message: str):
        # error级别的日志，红色
        formatted_message = "-" * 120 + '\n| ' + message + "\n" + "└" + "-" * 150
        LogColor.error_logger.error(formatted_message)
        print("\033[0;31m" + formatted_message + "\033[0m")

    @staticmethod
    def debug(message: str):
        # debug级别的日志，灰色
        LogColor.debug_logger.debug(message)
        print("\033[0;37m" + message + "\033[0m")