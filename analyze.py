#!/usr/bin/env python3

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from btm import btm_analysis
from centrality_score import centralityScore
from floodfire.common.logging import AppLogger
from floodfire.store.analyze_task import AnalyzeTaskStore


def centrality_analysis(task):
    logger.info("Starting centrality analysis...")

    stance = pd.DataFrame()
    raw_predict_data = pd.DataFrame()

    files = [f.name for f in Path(task_data_folder).iterdir() if f.is_file()]

    for file in files:
        # 找出開頭是 2 且結尾是 .csv 的檔案
        if file.endswith(".csv") and file.startswith("2"):
            print(file)
            # 取得檔案名稱（不含副檔名）作為日期格式
            formatted_date = file.split(".")[0]
            # 設定中心性分數儲存路徑
            score_csv_path = centrality_folder.joinpath(f"{formatted_date}.csv")

            if not score_csv_path.exists():
                # 讀取合併後的週資料
                combined_dataset = pd.read_csv(task_data_folder.joinpath(file))

                (result, filtered_dataset) = calc_centrality_scores(
                    formatted_date, score_csv_path, combined_dataset
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
                if stance.empty:
                    stance = temp_df
                else:
                    stance = pd.merge(stance, temp_df, on=["user", "name"], how="outer")
                # btm 主題模型分析並儲存
                calc_btm_topics(formatted_date, combined_dataset)

                # 合併原始立場預測資料
                if raw_predict_data.empty:
                    raw_predict_data = filtered_dataset
                else:
                    raw_predict_data = pd.merge(
                        raw_predict_data,
                        filtered_dataset,
                        on=["id"],
                        how="outer",
                        suffixes=("_raw", "_filtered"),
                    )
                    # 若有 prediction_filtered 欄位則移除並重新命名
                    if "prediction_filtered" in raw_predict_data.columns:
                        raw_predict_data.drop(
                            columns="prediction_filtered", inplace=True
                        )
                        raw_predict_data.rename(
                            columns={"prediction_raw": "prediction"}, inplace=True
                        )
    # 若有設定事件時間範圍則進行事件中心性分析
    if task["start_date_1"] and task["end_date_1"]:
        event_centrality_analysis(
            event_num=1,
            start=task["start_date_1"],
            end=task["end_date_1"],
            stance=stance,
            raw_predict_data=raw_predict_data,
        )

    if task["start_date_2"] and task["end_date_2"]:
        event_centrality_analysis(
            event_num=1,
            start=task["start_date_2"],
            end=task["end_date_2"],
            stance=stance,
            raw_predict_data=raw_predict_data,
        )

    if task["start_date_3"] and task["end_date_3"]:
        event_centrality_analysis(
            event_num=1,
            start=task["start_date_3"],
            end=task["end_date_3"],
            stance=stance,
            raw_predict_data=raw_predict_data,
        )

    # 將 stance DataFrame 中的缺失值填補為 "無資料"
    stance.fillna("無資料", inplace=True)

    # 依據欄位名稱是否為純數字排序 stance DataFrame 的欄位
    sorted_columns = sorted(stance.columns, key=lambda x: (x.isdigit(), x))
    stance = stance[sorted_columns]

    try:
        # 若 stance 資料夾不存在則建立，並將 stance DataFrame 儲存為 CSV 檔案
        stance_folder = task_data_folder.joinpath("stance")
        if not stance_folder.exists():
            stance_folder.mkdir(parents=True, exist_ok=True)
        stance_csv_path = stance_folder.joinpath("stance.csv")
        stance.to_csv(stance_csv_path, index=False)
        logger.info("Stance CSV file saved.")
    except Exception as e:
        logger.error(f"Error saving stance CSV: {e}")

    try:
        # 若 download 資料夾不存在則建立，並將 raw_predict_data 儲存為 CSV 檔案
        download_folder = task_data_folder.joinpath("download")
        if not download_folder.exists():
            download_folder.mkdir(parents=True, exist_ok=True)
        raw_predict_data.to_csv(
            download_folder.joinpath("raw_predict_data.csv"), index=False
        )
        logger.info("Download raw_predict_data CSV files saved.")
    except Exception as e:
        logger.error(f"Error saving download raw_predict_data CSV: {e}")


def event_centrality_analysis(event_num, start, end, stance, raw_predict_data):
    start_date = datetime.strptime(start, "%Y-%m-%d").strftime("%Y%m%d")
    end_date = datetime.strptime(end, "%Y-%m-%d").strftime("%Y%m%d")

    event_data_file_path = task_data_folder.joinpath(
        f"event{event_num}-{start_date}_{end_date}.csv"
    )
    event_score_csv_path = centrality_folder.joinpath(
        f"event{event_num}-{start_date}_{end_date}.csv"
    )
    (event_result, event_filtered_dataset) = calc_centrality_scores(
        f"event{event_num}-{start_date}_{end_date}",
        event_score_csv_path,
        event_data_file_path,
    )
    # 建立不同時間點的立場 DataFrame
    temp_df = pd.DataFrame(
        {
            "user": event_result["key"],
            "name": event_result["label"],
            f"event{event_num}-{start_date}_{end_date}": event_result["cluster"],
        }
    )
    if stance.empty:
        stance = temp_df
    else:
        stance = pd.merge(stance, temp_df, on=["user", "name"], how="outer")

    # btm 主題模型分析並儲存
    calc_btm_topics(f"event{event_num}-{start_date}_{end_date}", event_data_file_path)

    # 下載立場原始資料
    if raw_predict_data.empty:
        raw_predict_data = event_filtered_dataset
    else:
        raw_predict_data = pd.merge(
            raw_predict_data,
            event_filtered_dataset,
            on=["id"],
            how="outer",
            suffixes=("_raw", "_filtered"),
        )
        if "prediction_filtered" in raw_predict_data.columns:
            raw_predict_data.drop(columns="prediction_filtered", inplace=True)
            raw_predict_data.rename(
                columns={"prediction_raw": "prediction"}, inplace=True
            )


def calc_centrality_scores(prefix_filename, score_csv_path, combined_dataset):
    logger.info(f"Calculating centrality scores for [{prefix_filename}]")
    try:
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

        # 儲存各種中心性指標檔案
        save_centrality(
            prefix_filename,
            network_degree,
            network_betweenness,
            network_closeness,
            network_eigenvector,
        )
        logger.info(f"Centrality scores calculated and saved for [{prefix_filename}]")
    except Exception as e:
        logger.error(f"Error calculating [{prefix_filename}] centrality scores: {e}")
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
        logger.info(f"Centrality JSON files saved for [{prefix_name}]")
    except Exception as e:
        logger.error(f"Error saving centrality JSON files: {e}")


def calc_btm_topics(prefix_filename, combined_dataset):
    logger.info(f"Calculating BTM topics for [{prefix_filename}]")
    try:
        topics_coords_csv_path = topics_coords_folder.joinpath(
            f"{prefix_filename}_topics_coords.csv"
        )
        dict_file_path = task_data_folder.joinpath("dictionary.txt")
        if not topics_coords_csv_path.exists():
            if not dict_file_path.exists():
                # 沒有字典檔案的 BTM 分析
                topics_coords, terms_probs, top_5_doc = btm_analysis(combined_dataset)
            else:
                # 有字典檔案的 BTM 分析
                topics_coords, terms_probs, top_5_doc = btm_analysis(
                    combined_dataset, dict_file_path
                )

            # 儲存 BTM 分析結果
            topics_coords.to_csv(topics_coords_csv_path, index=False)
            logger.info(f"BTM topics coords saved: {topics_coords_csv_path}")
            # 儲存各主題詞機率
            for topic, df in terms_probs.items():
                terms_probs_csv_path = terms_probs_folder.joinpath(
                    f"{prefix_filename}_{topic}.csv"
                )
                df.to_csv(terms_probs_csv_path, index=False)
                logger.info(f"Terms probabilities saved: {terms_probs_csv_path}")

            # 儲存各主題前五篇文件
            for topic, df in top_5_doc.items():
                top_5_docs_csv_path = top_5_docs_folder.joinpath(
                    f"{prefix_filename}_{topic}.csv"
                )
                df.to_csv(top_5_docs_csv_path, index=False)
                logger.info(f"Top 5 documents saved: {top_5_docs_csv_path}")
        logger.info(f"BTM topics calculated and saved for [{prefix_filename}]")
    except Exception as e:
        logger.error(f"Error in calculating BTM analysis: {e}")
        return


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
    task_data_folder = Path("data/{}".format(task["name"]))
    if not task_data_folder.exists():
        task_data_folder.mkdir(parents=True)

    # Create centrality folder if it doesn't exist
    centrality_folder = task_data_folder.joinpath("centrality")
    if not centrality_folder.exists():
        centrality_folder.mkdir(parents=True, exist_ok=True)

    # Create topics_coords folder if it doesn't exist
    topics_coords_folder = task_data_folder.joinpath("btm", "topics_coords")
    if not topics_coords_folder.exists():
        topics_coords_folder.mkdir(parents=True, exist_ok=True)

    # Create terms_probs folder if it doesn't exist
    terms_probs_folder = task_data_folder.joinpath("btm", "terms_probs")
    if not terms_probs_folder.exists():
        terms_probs_folder.mkdir(parents=True, exist_ok=True)

    # Create top_5_docs folder if it doesn't exist
    top_5_docs_folder = task_data_folder.joinpath("btm", "top_5_docs")
    if not top_5_docs_folder.exists():
        top_5_docs_folder.mkdir(parents=True, exist_ok=True)

    # Create network folder if it doesn't exist
    network_folder = task_data_folder.joinpath("network")
    if not network_folder.exists():
        network_folder.mkdir(parents=True, exist_ok=True)
    centrality_analysis(task)
