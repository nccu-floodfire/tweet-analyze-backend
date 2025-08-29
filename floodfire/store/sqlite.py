#!/usr/bin/env python3

import sqlite3

from floodfire.basis.abc import BaseRDB
from floodfire.common.logging import AppLogger


class FloodFireSQLite(BaseRDB):
    def __init__(self, db_path: str, filename: str, log_path: str):
        super().__init__(db_path, filename)

        self._logger = AppLogger(
            name="FloodFireSQLite", file_path=log_path, file_name="floodfire_sqlite"
        )

        self._connect()

    def _dict_factory(self, cursor, row) -> dict:
        rtn_d = {}
        for idx, col in enumerate(cursor.description):
            rtn_d[col[0]] = row[idx]
        return rtn_d

    def _connect(self) -> None:
        try:
            self._conn = sqlite3.connect(self.db_file)
            self._conn.row_factory = self._dict_factory

            self._logger.info("成功連接到 SQLite 資料庫")
        except Exception as e:
            self._logger.error(f"連接到 SQLite 資料庫失敗: {e}")

    def _disconnect(self) -> None:
        self._conn.close()
