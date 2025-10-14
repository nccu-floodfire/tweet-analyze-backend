#!/usr/bin/env python3

import sqlite3
from contextlib import closing

from floodfire.basis.abc import BaseRDB
from floodfire.common.logging import AppLogger


class FloodFireSQLite(BaseRDB):
    def __init__(self, db_path: str, database_name: str, log_path: str):
        super().__init__(database_name)
        self.db_file = f"{db_path}/{self.database_name}.db"
        self._setup_logging(log_path)

        self._connect()

    def _setup_logging(self, log_path: str) -> None:
        self._logger = AppLogger(
            name="FloodFireSQLite", file_path=log_path, file_name="floodfire_sqlite"
        )

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

    def execute_query_show(self, sql: str) -> dict:
        """
        執行查詢語句並返回結果

        Args:
            sql (str): SQL 字串

        Returns:
            dict: 資料內容
        """
        if not self._conn:
            self._connect()
        try:
            with closing(self._conn.cursor()) as cur:
                cur.execute(sql)
                result = cur.fetchone()
        except Exception as err:
            self._logger.error(
                "Exception: [{db_name}] show_data function error. {err_msg}. SQL Statment: {sql_stmt}".format(
                    db_name=self.database_name, err_msg=repr(err), sql_stmt=sql
                )
            )
            raise err
        return result

    def execute_dml(self, sql: str, params: tuple = ()) -> int:
        """
        執行 DML 語句

        Args:
            sql (str): SQL 字串
        Returns:
            int: 受影響的行數
        """
        if not self._conn:
            self._connect()
        try:
            with closing(self._conn.cursor()) as cur:
                cur.execute(sql, params)
                self._conn.commit()
                affected_rows = cur.rowcount
                self._logger.info(
                    "Execute DML statement affected {row_count} rows.".format(
                        row_count=affected_rows
                    )
                )
                return affected_rows
        except Exception as err:
            self._logger.error(
                "Exception: [{db_name}] execute_dml function error. {err_msg}. SQL Statment: {sql_stmt}".format(
                    db_name=self.database_name, err_msg=repr(err), sql_stmt=sql
                )
            )
            raise err
