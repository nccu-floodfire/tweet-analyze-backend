#!/usr/bin/env python3

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from floodfire.basis.abc import BaseLogger


class AppLogger(BaseLogger):
    """
    一個實現了 Singleton 模式的日誌類別。
    無論您嘗試建立多少次實例，對於同一個 name，永遠只會回傳第一個建立的實例。
    """

    def _configure_logger(self):
        # 這段程式碼現在只會在每個 name 第一次建立實例時被呼叫一次
        if not self.logger.handlers:
            log_format = logging.Formatter(
                "%(asctime)s - %(name)12s - %(levelname)8s - [%(filename)s:%(lineno)d] - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(log_format)
            self.logger.addHandler(console_handler)

            # 如果有指定檔案路徑和檔名，則建立檔案處理器
            if self.file_path and self.file_name:
                # 檢查目錄是否存在，不存在則建立
                check_path = Path(self.file_path)
                if not check_path.exists():
                    check_path.mkdir(parents=True, exist_ok=True)

                # 組合完整的日誌檔案路徑
                full_log_path = "{filepath}/{filename}.log".format(
                    filepath=self.file_path, filename=self.file_name
                )

                # 建立一個將日誌寫入檔案的處理器
                file_handler = RotatingFileHandler(
                    full_log_path, maxBytes=1048576, backupCount=5
                )
                file_handler.setFormatter(log_format)
                self.logger.addHandler(file_handler)

            # 防止日誌訊息傳播到 root logger
            self.logger.propagate = False
