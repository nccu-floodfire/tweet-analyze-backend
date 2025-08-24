#!/usr/bin/env python3

import logging
from abc import ABC, abstractmethod


class BaseLogger(ABC):
    """
    一個抽象的日誌基底類別，定義了日誌記錄器的基本結構。
    子類別必須實作 _configure_logger 方法來設定日誌記錄器。
    """

    _instances = {}

    def __new__(cls, name, level=logging.INFO, file_path=None, file_name=None):
        """
        使用單例模式，確保同一個 name 的 Logger 只會被設定一次。
        """
        if name not in cls._instances:
            # 如果這個 name 的實例不存在，則建立一個新的
            instance = super().__new__(cls)
            cls._instances[name] = instance
        # 回傳已存在的實例
        return cls._instances[name]

    def __init__(self, name, level="INFO", file_path=None, file_name=None):
        # 檢查是否已經初始化過，防止重複設定
        if hasattr(self, "logger"):
            return
        self.logger = logging.getLogger(name)
        self._set_level(level)
        self.file_path = file_path
        self.file_name = file_name

        self._configure_logger()

    @abstractmethod
    def _configure_logger(self):
        """
        設定日誌記錄器的格式和處理器。
        子類別必須實作這個方法。
        """
        pass

    def _set_level(self, level: str):
        """
        設定 log 層級

        Args:
            level (str): log 層級
        """
        log_level = {
            "NOTSET": 0,
            "DEBUG": 10,
            "INFO": 20,
            "WARNING": 30,
            "ERROR": 40,
            "CRITICAL": 50,
        }
        if level in log_level:
            self.logger.setLevel(log_level[level])
        else:
            raise ValueError("The given log level is empty!")

    def debug(self, msg, *args, **kwargs):
        self.logger.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        self.logger.critical(msg, *args, **kwargs)
