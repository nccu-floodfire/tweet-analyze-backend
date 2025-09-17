#!/usr/bin/env python3

from datetime import datetime

from floodfire.store.sqlite import FloodFireSQLite


class AnalyzeTaskStore:
    def __init__(self, db_path: str, log_path: str):
        self._db = FloodFireSQLite(db_path, "tweet_analyze", log_path)

    def store_new_task(self, task_data: dict) -> int:
        sql = "\
        INSERT INTO tasks (name, start_date_1, end_date_1, start_date_2, end_date_2, start_date_3, end_date_3, datasets_list_len)\
        VALUES (?, ?, ?, ?, ?, ?, ?, ?);"
        params = (
            task_data["name"],
            task_data.get("start_date_1", None),
            task_data.get("end_date_1", None),
            task_data.get("start_date_2", None),
            task_data.get("end_date_2", None),
            task_data.get("start_date_3", None),
            task_data.get("end_date_3", None),
            task_data.get("datasets_list_len", None),
        )
        return self._db.execute_dml(sql, params)

    def get_last_task(self):
        sql = "SELECT * FROM tasks\
            WHERE centrality = 0 OR topics_coords = 0\
            OR terms_probs = 0 OR top_5_doc = 0 OR network = 0\
            ORDER BY created_at DESC LIMIT 0,1;"

        return self._db.execute_query_show(sql)

    def update_task_phase(self, task_id: int, phase_name: str):
        # now_time = datetime.now()
        sql = f"UPDATE tasks SET {phase_name} = {phase_name} + 1 WHERE id = ?;"
        params = (task_id,)
        return self._db.execute_dml(sql, params)
