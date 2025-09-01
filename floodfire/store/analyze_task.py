#!/usr/bin/env python3

from floodfire.store.sqlite import FloodFireSQLite


class AnalyzeTaskStore:
    def __init__(self, db_path: str, log_path: str):
        self._db = FloodFireSQLite(db_path, "tweet_analyze", log_path)

    def store_new_task(self, task_data: dict) -> int:
        sql = "\
        INSERT INTO tasks (name, start_date_1, end_date_1, start_date_2, end_date_2, start_date_3, end_date_3)\
        VALUES (?, ?, ?, ?, ?, ?, ?);"
        params = (
            task_data["name"],
            task_data.get("start_date_1", None),
            task_data.get("end_date_1", None),
            task_data.get("start_date_2", None),
            task_data.get("end_date_2", None),
            task_data.get("start_date_3", None),
            task_data.get("end_date_3", None),
        )
        return self._db.execute_dml(sql, params)
