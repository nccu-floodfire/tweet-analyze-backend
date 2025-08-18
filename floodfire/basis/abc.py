#!/usr/bin/env python3

import logging
from abc import ABC, ABCMeta, abstractmethod


class BaseLogger(metaclass=ABCMeta):
    _msg_format = "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"
    _date_format = "%Y-%m-%d %H:%M:%S"
    _logger_name = ""
    _file_path = ""

    @property
    @abstractmethod
    def logger_name(self):
        return self._logger_name

    @logger_name.setter
    @abstractmethod
    def logger_name(self, logger_name):
        self._logger_name = logger_name

    @property
    @abstractmethod
    def file_path(self):
        return self._file_path

    @file_path.setter
    @abstractmethod
    def file_path(self, file_path=None):
        self._file_path = file_path

    @abstractmethod
    def log(self, message: str):
        pass
