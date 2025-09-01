#!/usr/bin/env python3

from pathlib import Path

from floodfire.common.logging import AppLogger
from floodfire.store.analyze_task import AnalyzeTaskStore


def centrality_analysis(task):
    # Perform centrality analysis on the task
    task_name = task["name"]
    data_save_path = Path("data/{}".format(task_name))
    centrality_folder = data_save_path.joinpath("centrality")
    if not centrality_folder.exists():
        centrality_folder.mkdir()


def topics_coords_analysis(task):
    # Perform topics coordinates analysis on the task
    task_name = task["name"]
    data_save_path = Path("data/{}".format(task_name))
    topics_coords_folder = data_save_path.joinpath("btm", "topics_coords")
    if not topics_coords_folder.exists():
        topics_coords_folder.mkdir()


def terms_probs_analysis(task):
    # Perform terms probabilities analysis on the task
    task_name = task["name"]
    data_save_path = Path("data/{}".format(task_name))
    terms_probs_folder = data_save_path.joinpath("btm", "terms_probs")
    if not terms_probs_folder.exists():
        terms_probs_folder.mkdir()


def top_5_doc_analysis(task):
    # Perform top 5 documents analysis on the task
    task_name = task["name"]
    data_save_path = Path("data/{}".format(task_name))
    top_5_docs_folder = data_save_path.joinpath("btm", "top_5_docs")
    if not top_5_docs_folder.exists():
        top_5_docs_folder.mkdir()


def network_analysis(task):
    # Perform network analysis on the task
    task_name = task["name"]
    data_save_path = Path("data/{}".format(task_name))
    network_folder = data_save_path.joinpath("network")
    if not network_folder.exists():
        network_folder.mkdir()


if __name__ == "__main__":
    # Initialize directory paths
    dir_path = Path(__file__).resolve().parent

    # Initialize logger
    logger = AppLogger("analyze", file_path="logs", file_name="analyze")
    logger.info("Starting analysis...")

    # Initialize task store
    db_folder = "{}/db".format(dir_path)
    task_store = AnalyzeTaskStore(db_folder, "logs/")

    task = task_store.get_last_task()
    print(task)

    if not task:
        logger.warning("No task found.")
