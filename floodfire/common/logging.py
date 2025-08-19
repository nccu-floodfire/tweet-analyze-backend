#!/usr/bin/env python3

import logging
import sys

from floodfire.basis.abc import BaseLogger


class AppLogger(BaseLogger):
    """
    一個實現了 Singleton 模式的日誌類別。
    無論您嘗試建立多少次實例，對於同一個 name，永遠只會回傳第一個建立的實例。
    """

    _instances = {}

    def __new__(cls, name, level=logging.INFO):
        if name not in cls._instances:
            # 如果這個 name 的實例不存在，則建立一個新的
            instance = super().__new__(cls)
            cls._instances[name] = instance
        # 回傳已存在的實例
        return cls._instances[name]

    def _configure_logger(self):
        # 這段程式碼現在只會在每個 name 第一次建立實例時被呼叫一次
        if not self.logger.handlers:
            log_format = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
            )
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(log_format)
            self.logger.addHandler(console_handler)

            file_handler = logging.FileHandler("app.log")
            file_handler.setFormatter(log_format)
            self.logger.addHandler(file_handler)

            self.logger.propagate = False
