#!/usr/bin/env python3

import logging
from abc import ABC, abstractmethod


class BaseLogger(ABC):
    """
    一個抽象的日誌基底類別，定義了日誌記錄器的基本結構。
    子類別必須實作 _configure_logger 方法來設定日誌記錄器。
    """

    def __init__(self, name, level=logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self._configure_logger()

    @abstractmethod
    def _configure_logger(self):
        """
        設定日誌記錄器的格式和處理器。
        子類別必須實作這個方法。
        """
        pass

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
