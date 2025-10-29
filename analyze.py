#!/usr/bin/env python3

import csv
import json
import zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd

from btm import btm_analysis
from centrality_score import centralityScore
from floodfire.common.logging import AppLogger
from floodfire.store.analyze_task import AnalyzeTaskStore


def centrality_analysis(task):
    """
    分析中心性相關的主程式

    Args:
        task (dict): 任務內容
    """
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
        raw_predict_file = download_folder.joinpath("raw_predict_data.csv")
        if not download_folder.exists():
            download_folder.mkdir(parents=True, exist_ok=True)
        raw_predict_data.to_csv(
            raw_predict_file, index=False
        )
        logger.info("Download raw_predict_data CSV files saved.")
        # 壓縮預測資料檔案並刪除原始檔案
        zip_predict_file(download_folder, raw_predict_file)
    except Exception as e:
        logger.error(f"Error saving download raw_predict_data CSV: {e}")


def event_centrality_analysis(event_num, start, end, stance, raw_predict_data):
    """
    事件類型的在中心性分析

    Args:
        event_num (int): 事件編號
        start (str): 事件開始時間
        end (str): 事件結束時間
        stance (DataFrame): 事件立場資料
        raw_predict_data (DataFrame): 原始預測資料
    """
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
    """
    計算中心性分數

    Args:
        prefix_filename (str): 檔案名稱前綴
        score_csv_path (str): 中心性分數 CSV 檔案路徑
        combined_dataset (DataFrame): 合併後的資料集

    Returns:
        _type_: _description_
    """
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
        # 更新任務階段為中心性計算完成
        task_store.update_task_phase(task["id"], "centrality")

        # 儲存各種中心性指標檔案
        save_centrality_json(
            prefix_filename,
            network_degree,
            network_betweenness,
            network_closeness,
            network_eigenvector,
        )

        save_centrality_csv(
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


def save_centrality_json(
    prefix_name: str,
    network_degree: dict,
    network_betweenness: dict,
    network_closeness: dict,
    network_eigenvector: dict,
):
    """
    儲存各種中心性指標

    Args:
        prefix_name (string): 檔案名稱前綴
        network_degree (Dict): 節點的度數中心性
        network_betweenness (Dict): 節點的介數中心性
        network_closeness (Dict): 節點的接近中心性
        network_eigenvector (Dict): 節點的特徵向量中心性
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
        # 更新任務階段為網路分析完成
        task_store.update_task_phase(task["id"], "network")
    except Exception as e:
        logger.error(f"Error saving centrality JSON files: {e}")


def save_centrality_csv(
    prefix_name: str,
    network_degree: dict,
    network_betweenness: dict,
    network_closeness: dict,
    network_eigenvector: dict,
):
    """
    儲存各種中心性指標 csv 檔案

    Args:
        prefix_name (string): 檔案名稱前綴
        network_degree (Dict): 節點的度數中心性
        network_betweenness (Dict): 節點的介數中心性
        network_closeness (Dict): 節點的接近中心性
        network_eigenvector (Dict): 節點的特徵向量中心性
    """
    try:
        # 儲存各種中心性指標的 csv 檔案
        download_folder = task_data_folder.joinpath("download")
        # Gephi 匯出資料夾，若不存在則建立
        gephi_folder = download_folder.joinpath("gephi")
        if not gephi_folder.exists():
            gephi_folder.mkdir(parents=True, exist_ok=True)

        degree_nodes_csv = gephi_folder.joinpath(f"{prefix_name}_degree_nodes.csv")
        degree_edges_csv = gephi_folder.joinpath(f"{prefix_name}_degree_edges.csv")
        betweenness_nodes_csv = gephi_folder.joinpath(
            f"{prefix_name}_betweenness_nodes.csv"
        )
        betweenness_edges_csv = gephi_folder.joinpath(
            f"{prefix_name}_betweenness_edges.csv"
        )
        closeness_nodes_csv = gephi_folder.joinpath(
            f"{prefix_name}_closeness_nodes.csv"
        )
        closeness_edges_csv = gephi_folder.joinpath(
            f"{prefix_name}_closeness_edges.csv"
        )
        eigenvector_nodes_csv = gephi_folder.joinpath(
            f"{prefix_name}_eigenvector_nodes.csv"
        )
        eigenvector_edges_csv = gephi_folder.joinpath(
            f"{prefix_name}_eigenvector_edges.csv"
        )

        node_fieldnames = ["id", "label", "tag", "cluster", "score"]
        edge_fieldnames = ["source", "target", "type"]

        # 轉換 network degree node 的 key 名稱給 Gephi 使用
        network_degree_nodes = transform_gephi_nodes(network_degree["nodes"])
        with open(degree_nodes_csv, "w", newline="") as csv_file:
            # Create a DictWriter object
            writer = csv.DictWriter(
                csv_file, fieldnames=node_fieldnames, extrasaction="ignore"
            )
            # Write the header row
            writer.writeheader()
            # Write the data rows
            writer.writerows(network_degree_nodes)

        # 轉換 network degree edge 的 key 名稱給 Gephi 使用
        network_degree_edges = transform_gephi_edges(network_degree["edges"])
        with open(degree_edges_csv, "w", newline="") as csv_file:
            # Create a DictWriter object
            writer = csv.DictWriter(
                csv_file, fieldnames=edge_fieldnames, extrasaction="ignore"
            )
            # Write the header row
            writer.writeheader()
            # Write the data rows
            writer.writerows(network_degree_edges)

        # 轉換 network betweenness node 的 key 名稱給 Gephi 使用
        network_betweenness_nodes = transform_gephi_nodes(network_betweenness["nodes"])
        with open(betweenness_nodes_csv, "w", newline="") as csv_file:
            # Create a DictWriter object
            writer = csv.DictWriter(
                csv_file, fieldnames=node_fieldnames, extrasaction="ignore"
            )
            # Write the header row
            writer.writeheader()
            # Write the data rows
            writer.writerows(network_betweenness_nodes)

        # 轉換 network betweenness edge 的 key 名稱給 Gephi 使用
        network_betweenness_edges = transform_gephi_edges(network_betweenness["edges"])
        with open(betweenness_edges_csv, "w", newline="") as csv_file:
            # Create a DictWriter object
            writer = csv.DictWriter(
                csv_file, fieldnames=edge_fieldnames, extrasaction="ignore"
            )
            # Write the header row
            writer.writeheader()
            # Write the data rows
            writer.writerows(network_betweenness_edges)

        # 轉換 network closeness node 的 key 名稱給 Gephi 使用
        network_closeness_nodes = transform_gephi_nodes(network_closeness["nodes"])
        with open(closeness_nodes_csv, "w", newline="") as csv_file:
            # Create a DictWriter object
            writer = csv.DictWriter(
                csv_file, fieldnames=node_fieldnames, extrasaction="ignore"
            )
            # Write the header row
            writer.writeheader()
            # Write the data rows
            writer.writerows(network_closeness_nodes)

        # 轉換 network closeness edge 的 key 名稱給 Gephi 使用
        network_closeness_edges = transform_gephi_edges(network_closeness["edges"])
        with open(closeness_edges_csv, "w", newline="") as csv_file:
            # Create a DictWriter object
            writer = csv.DictWriter(
                csv_file, fieldnames=edge_fieldnames, extrasaction="ignore"
            )
            # Write the header row
            writer.writeheader()
            # Write the data rows
            writer.writerows(network_closeness_edges)

        # 轉換 network eigenvector node 的 key 名稱給 Gephi 使用
        network_eigenvector_nodes = transform_gephi_nodes(network_eigenvector["nodes"])
        with open(eigenvector_nodes_csv, "w", newline="") as csv_file:
            # Create a DictWriter object
            writer = csv.DictWriter(
                csv_file, fieldnames=node_fieldnames, extrasaction="ignore"
            )
            # Write the header row
            writer.writeheader()
            # Write the data rows
            writer.writerows(network_eigenvector_nodes)

        # 轉換 network eigenvector edge 的 key 名稱給 Gephi 使用
        network_eigenvector_edges = transform_gephi_edges(network_eigenvector["edges"])
        with open(eigenvector_edges_csv, "w", newline="") as csv_file:
            # Create a DictWriter object
            writer = csv.DictWriter(
                csv_file, fieldnames=edge_fieldnames, extrasaction="ignore"
            )
            # Write the header row
            writer.writeheader()
            # Write the data rows
            writer.writerows(network_eigenvector_edges)

        zip_gephi_files(download_folder)

        # 記錄成功訊息
        logger.info(f"Centrality CSV files saved for [{prefix_name}]")
    except Exception as e:
        logger.error(f"Error saving centrality CSV files: {e}")


def zip_gephi_files(download_folder: Path):
    """
    將 download 的 gephi 資料夾壓縮成 zip 壓縮檔

    Args:
        download_folder (Path): Gephi 資料夾路徑
    """
    try:
        gephi_folder = download_folder.joinpath("gephi")
        zip_filename = download_folder.joinpath("for-gephi.zip")

        if gephi_folder.exists() and gephi_folder.is_dir():
            with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as my_zip:
                for file_path in gephi_folder.iterdir():
                    if file_path.is_file():
                        # 將檔案加入 zip，arcname 只保留檔名
                        my_zip.write(file_path, arcname=file_path.name)

            logger.info(f"Gephi files zipped: {zip_filename}")
    except Exception as e:
        logger.error(f"Error zipping Gephi files: {e}")

def zip_predict_file(download_folder: Path, predict_file: Path):
    """
    將 download 的預測資料壓縮成 zip 壓縮檔
    完成壓縮後刪除原始被壓縮的檔案

    Args:
        download_folder (Path): download 資料夾路徑
        predict_file (Path): 欲壓縮的檔案路徑
    """
    try:
        zip_filename = download_folder.joinpath("raw_predict_data.zip")

        if predict_file.exists() and predict_file.is_file():
            with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as my_zip:
                # 將檔案加入 zip，arcname 只保留檔名
                my_zip.write(predict_file, arcname=predict_file.name)

            logger.info(f"Predict file zipped: {zip_filename}")
            # 刪除原始 raw_predict_data.csv 檔案
            predict_file.unlink()
        else:
            logger.warning(f"Predict file not found: {predict_file}")
    except Exception as e:
        logger.error(f"Error zipping predict file: {e}")


def transform_gephi_nodes(nodes):
    transformed_data = []
    for node in nodes:
        transformed_node = {
            "id": node["key"],
            "label": node["label"],
            "tag": node["tag"],
            "cluster": node["cluster"],
            "score": node["score"],
        }
        transformed_data.append(transformed_node)
    return transformed_data


def transform_gephi_edges(edges):
    transformed_data = []
    for edge in edges:
        transformed_edge = {
            "source": edge[0],
            "target": edge[1],
            "type": "Directed",
        }
        transformed_data.append(transformed_edge)
    return transformed_data


def calc_btm_topics(prefix_filename, combined_dataset):
    """
    計算 BTM 主題

    Args:
        prefix_filename (str): 檔案名稱前綴
        combined_dataset (DataFrame): 合併後的資料集
    """
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
            # 更新任務階段為主題模型完成
            task_store.update_task_phase(task["id"], "topics_coords")

            # 儲存各主題詞機率
            for topic, df in terms_probs.items():
                terms_probs_csv_path = terms_probs_folder.joinpath(
                    f"{prefix_filename}_{topic}.csv"
                )
                df.to_csv(terms_probs_csv_path, index=False)
                logger.info(f"Terms probabilities saved: {terms_probs_csv_path}")
            # 更新任務階段為主題詞機率完成
            task_store.update_task_phase(task["id"], "terms_probs")

            # 儲存各主題前五篇文件
            for topic, df in top_5_doc.items():
                top_5_docs_csv_path = top_5_docs_folder.joinpath(
                    f"{prefix_filename}_{topic}.csv"
                )
                df.to_csv(top_5_docs_csv_path, index=False)
                logger.info(f"Top 5 documents saved: {top_5_docs_csv_path}")
            # 更新任務階段為前五篇文件完成
            task_store.update_task_phase(task["id"], "top_5_doc")

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
    task_data_folder = Path("{}/data/{}".format(dir_path, task["name"]))
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
    # Create download folder if it doesn't exist
    download_folder = task_data_folder.joinpath("download")
    if not download_folder.exists():
        download_folder.mkdir(parents=True, exist_ok=True)

    centrality_analysis(task)
