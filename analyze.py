#!/usr/bin/env python3

import json
from pathlib import Path

import pandas as pd

from centrality_score import centralityScore
from floodfire.common.logging import AppLogger
from floodfire.store.analyze_task import AnalyzeTaskStore


def centrality_analysis(task):
    logger.info("Starting centrality analysis...")

    Stance = pd.DataFrame()
    raw_predict_data = pd.DataFrame()

    files = [f.name for f in Path(data_folder).iterdir() if f.is_file()]

    for file in files:
        # 找出開頭是 2 且結尾是 .csv 的檔案
        if file.endswith(".csv") and file.startswith("2"):
            print(file)
            # 取得檔案名稱（不含副檔名）作為日期格式
            formatted_date = file.split(".")[0]
            # 設定中心性分數儲存路徑
            score_csv_path = centrality_folder.joinpath(f"{formatted_date}.csv")

            if not score_csv_path.exists():
                (result, filtered_dataset) = calc_centrality_scores(
                    file, score_csv_path
                )
                print(result)
                # 建立不同時間點的立場 DataFrame
                temp_df = pd.DataFrame(
                    {
                        "user": result["key"],
                        "name": result["label"],
                        f"{formatted_date}": result["cluster"],
                    }
                )
                # 合併立場資料
                if Stance.empty:
                    Stance = temp_df
                else:
                    Stance = pd.merge(Stance, temp_df, on=["user", "name"], how="outer")


def calc_centrality_scores(file, score_csv_path):
    # 讀取合併後的週資料
    combined_dataset = pd.read_csv(data_folder.joinpath(file))
    (
        score,
        network_degree,
        network_betweenness,
        network_closeness,
        network_eigenvector,
        result,
        filtered_dataset,
    ) = centralityScore(combined_dataset, score_csv_path)
    # 儲存中心性分數 CSV
    score.to_csv(score_csv_path, index=False)
    # 取得檔案名稱（不含副檔名）作為日期格式
    formatted_date = file.split(".")[0]
    # 儲存各種中心性指標檔案
    save_centrality(
        formatted_date,
        network_degree,
        network_betweenness,
        network_closeness,
        network_eigenvector,
    )
    return result, filtered_dataset


def save_centrality(
    prefix_name,
    network_degree,
    network_betweenness,
    network_closeness,
    network_eigenvector,
):
    """
    儲存各種中心性指標

    Args:
        prefix_name (string): 檔案名稱前綴
        network_degree (DataFrame): 節點的度數中心性
        network_betweenness (DataFrame): 節點的介數中心性
        network_closeness (DataFrame): 節點的接近中心性
        network_eigenvector (DataFrame): 節點的特徵向量中心性
    """
    try:
        # 儲存各種中心性指標的 JSON 檔案
        degree_json = network_folder.joinpath(f"{prefix_name}_degree.json")
        betweenness_json = network_folder.joinpath(f"{prefix_name}_betweenness.json")
        closeness_json = network_folder.joinpath(f"{prefix_name}_closeness.json")
        eigenvector_json = network_folder.joinpath(f"{prefix_name}_eigenvector.json")

        with open(degree_json, "w") as file:
            json.dump(network_degree, file, indent=4)
        with open(betweenness_json, "w") as file:
            json.dump(network_betweenness, file, indent=4)
        with open(closeness_json, "w") as file:
            json.dump(network_closeness, file, indent=4)
        with open(eigenvector_json, "w") as file:
            json.dump(network_eigenvector, file, indent=4)
        # 記錄成功訊息
        logger.info(f"Centrality JSON files saved for {prefix_name}")
    except Exception as e:
        logger.error(f"Error saving centrality JSON files: {e}")


def calc_btm_topics(task):
    pass


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

    # 建立分析所需的資料夾
    logger.info("Creating analysis folders...")
    # Create the main data folder if it doesn't exist
    data_folder = Path("data/{}".format(task["name"]))
    if not data_folder.exists():
        data_folder.mkdir(parents=True)

    # Create centrality folder if it doesn't exist
    centrality_folder = data_folder.joinpath("centrality")
    if not centrality_folder.exists():
        centrality_folder.mkdir(parents=True, exist_ok=True)

    # Create topics_coords folder if it doesn't exist
    topics_coords_folder = data_folder.joinpath("btm", "topics_coords")
    if not topics_coords_folder.exists():
        topics_coords_folder.mkdir(parents=True, exist_ok=True)

    # Create terms_probs folder if it doesn't exist
    terms_probs_folder = data_folder.joinpath("btm", "terms_probs")
    if not terms_probs_folder.exists():
        terms_probs_folder.mkdir(parents=True, exist_ok=True)

    # Create top_5_docs folder if it doesn't exist
    top_5_docs_folder = data_folder.joinpath("btm", "top_5_docs")
    if not top_5_docs_folder.exists():
        top_5_docs_folder.mkdir(parents=True, exist_ok=True)

    # Create network folder if it doesn't exist
    network_folder = data_folder.joinpath("network")
    if not network_folder.exists():
        network_folder.mkdir(parents=True, exist_ok=True)
    centrality_analysis(task)
