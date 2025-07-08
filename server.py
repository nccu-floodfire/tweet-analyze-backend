import glob
import json
import os
import re
from datetime import datetime, timedelta

import pandas as pd
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS  # Import CORS from Flask-CORS

from btm import btm_analysis
from centerityScore import centerityScore
from network import network
from statisticCalcu import statisticCalcu

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes


@app.route("/upload", methods=["POST"])
def upload():  # 儲存檔案及分週檔案
    file = request.files["file"]
    #     print(file)

    start_date1 = request.form.get("startDate1")  # 獲取開始日期
    end_date1 = request.form.get("endDate1")  # 獲取結束日期
    start_date2 = request.form.get("startDate2")  # 獲取開始日期
    end_date2 = request.form.get("endDate2")  # 獲取結束日期
    start_date3 = request.form.get("startDate3")  # 獲取開始日期
    end_date3 = request.form.get("endDate3")  # 獲取結束日期

    new_base_name = f"{file.filename}_{start_date1}_{end_date1}_{start_date2}_{end_date2}_{start_date3}_{end_date3}"
    new_filename = f"{new_base_name}.csv"
    folder_name = new_base_name

    # 放置資料的資料夾
    data_folder = "data"
    # 將上傳的檔案儲存成新的檔案名稱
    file.save(os.path.join(data_folder, new_filename))

    # 建立事件資料夾
    # EX：event_folder 可能為 "data/yourfile_20230101_20230107_20230108_20230114_20230115_20230121"
    event_folder = os.path.join(data_folder, folder_name)
    #     print(event_folder)
    if not os.path.exists(event_folder):
        os.makedirs(event_folder)
    #     if os.path.exists(os.path.join(data_folder,new_filename)):
    #         return jsonify({'result': '上傳成功1'})

    # 讀取剛剛儲存的 CSV 檔案，指定使用的引擎與引號字元以避免解析錯誤
    file = pd.read_csv(
        os.path.join(data_folder, new_filename),
        engine="python",
        quotechar='"',
    )

    # 將 "created_at" 欄位轉換為 datetime 物件，方便後續依時間分組與處理
    file["created_at"] = pd.to_datetime(file["created_at"])

    # 將 "created_at" 欄位依每週分組，新增 "period" 欄位
    file["period"] = file["created_at"].dt.to_period("W-SUN")
    groups = file.groupby("period")

    # 將每週的資料分組存入字典，key 為週期，value 為該週的資料
    datasets = {}
    for period, group in groups:
        datasets[period] = group

    # 將分組後的資料轉為 list，方便後續合併處理
    datasets_list = list(datasets.values())
    print(datasets_list)
    print(len(datasets_list))

    # 依據每兩週合併資料，並將合併後的資料儲存為新的 CSV 檔案
    for i in range(len(datasets_list) - 1):
        # 合併連續兩週的資料
        combined_dataset = pd.concat([datasets_list[i], datasets_list[i + 1]])
        # 取得合併後資料的最大日期，作為檔案名稱的一部分
        max_date = combined_dataset["created_at"].max().strftime("%Y%m%d")
        print(max_date)
        # 將合併後的資料儲存為 CSV 檔案，檔名為最大日期
        combined_dataset.to_csv(
            os.path.join(event_folder, f"{max_date}.csv"), index=False
        )

    # 根據使用者輸入的事件一日期範圍，篩選資料並儲存為對應的 CSV 檔案
    if start_date1 and end_date1:
        start_date1 = datetime.strptime(start_date1, "%Y-%m-%d")
        start_date1 = start_date1.strftime("%Y%m%d")
        end_date1 = datetime.strptime(end_date1, "%Y-%m-%d")
        end_date1 = end_date1.strftime("%Y%m%d")

        data = file[file["created_at"].between(start_date1, end_date1)]
        data.to_csv(
            os.path.join(event_folder, f"事件一：{start_date1}_{end_date1}.csv"),
            index=False,
        )
    # 根據使用者輸入的事件二日期範圍，篩選資料並儲存為對應的 CSV 檔案
    if start_date2 and end_date2:
        start_date2 = datetime.strptime(start_date2, "%Y-%m-%d")
        start_date2 = start_date2.strftime("%Y%m%d")
        end_date2 = datetime.strptime(end_date2, "%Y-%m-%d")
        end_date2 = end_date2.strftime("%Y%m%d")

        data = file[file["created_at"].between(start_date2, end_date2)]
        data.to_csv(
            os.path.join(event_folder, f"事件二：{start_date2}_{end_date2}.csv"),
            index=False,
        )
    # 根據使用者輸入的事件三日期範圍，篩選資料並儲存為對應的 CSV 檔案
    if start_date3 and end_date3:
        start_date3 = datetime.strptime(start_date3, "%Y-%m-%d")
        start_date3 = start_date3.strftime("%Y%m%d")
        end_date3 = datetime.strptime(end_date3, "%Y-%m-%d")
        end_date3 = end_date3.strftime("%Y%m%d")

        data = file[file["created_at"].between(start_date3, end_date3)]
        data.to_csv(
            os.path.join(event_folder, f"事件三：{start_date3}_{end_date3}.csv"),
            index=False,
        )

    return jsonify({"result": "Success!"})


@app.route("/centerity", methods=["POST"])
def centerity():  # 儲存社群網路資料及btm資料
    file = request.files["file"]
    dic_file = request.files.get("dic_file")
    startDate1 = request.form.get("startDate1")  # 獲取開始日期
    endDate1 = request.form.get("endDate1")  # 獲取結束日期
    startDate2 = request.form.get("startDate2")  # 獲取開始日期
    endDate2 = request.form.get("endDate2")  # 獲取結束日期
    startDate3 = request.form.get("startDate3")  # 獲取開始日期
    endDate3 = request.form.get("endDate3")  # 獲取結束日期

    new_filename = f"{file.filename}_{startDate1}_{endDate1}_{startDate2}_{endDate2}_{startDate3}_{endDate3}"
    data_folder = "data"
    full_path = os.path.join(data_folder, new_filename)

    save_folder = new_filename
    folder2 = "centerity"
    folder3 = "btm"
    folder4 = "topics_coords"
    folder5 = "terms_probs"
    folder6 = "top_5_doc"
    folder7 = "network"

    save_path = os.path.join(save_folder, folder2)
    save_path2 = os.path.join(save_folder, folder3, folder4)
    save_path3 = os.path.join(save_folder, folder3, folder5)
    save_path4 = os.path.join(save_folder, folder3, folder6)
    save_path5 = os.path.join(save_folder, folder7)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(save_path2):
        os.makedirs(save_path2)
    if not os.path.exists(save_path3):
        os.makedirs(save_path3)
    if not os.path.exists(save_path4):
        os.makedirs(save_path4)
    if not os.path.exists(save_path5):
        os.makedirs(save_path5)

    # save dic_file to os.path.join(save_folder,dictionary.txt)
    if dic_file:
        dic_path = os.path.join(save_folder, "dictionary.txt")
        dic_file.save(dic_path)

    files = os.listdir(full_path)
    print(files)
    Stance = pd.DataFrame()
    raw_predict_data = pd.DataFrame()

    for file in files:
        if file.startswith("2"):
            print(file)
            formatted_date = file.split(".")[0]
            path = os.path.join(save_path, f"{formatted_date}.csv")
            degree_path = os.path.join(save_path5, f"{formatted_date}_degree.json")
            betweeness_path = os.path.join(
                save_path5, f"{formatted_date}_betweeness.json"
            )
            closeness_path = os.path.join(
                save_path5, f"{formatted_date}_closeness.json"
            )
            eigenvector_path = os.path.join(
                save_path5, f"{formatted_date}_eigenvector.json"
            )

            if not os.path.exists(path):
                combined_dataset = pd.read_csv(os.path.join(full_path, file))
                (
                    score,
                    network_degree,
                    network_betweeness,
                    network_closeness,
                    network_eigenvector,
                    result,
                    filtered_dataset,
                ) = centerityScore(combined_dataset, path)
                # score = centerityScore(combined_dataset, path)

                if not os.path.exists(path):  # 中心性
                    score.to_csv(path, index=False)
                with open(degree_path, "w") as file:
                    json.dump(network_degree, file, indent=4)
                with open(betweeness_path, "w") as file:
                    json.dump(network_betweeness, file, indent=4)
                with open(closeness_path, "w") as file:
                    json.dump(network_closeness, file, indent=4)
                with open(eigenvector_path, "w") as file:
                    json.dump(network_eigenvector, file, indent=4)

                # 不同時間點立場
                temp_df = pd.DataFrame(
                    {
                        "user": result["key"],
                        "name": result["label"],
                        f"{formatted_date}": result["cluster"],
                    }
                )
                if Stance.empty:
                    Stance = temp_df
                else:
                    Stance = pd.merge(Stance, temp_df, on=["user", "name"], how="outer")
                # btm
                path1 = os.path.join(save_path2, f"{formatted_date}.csv")
                if not os.path.exists(path1):
                    if dic_file:
                        topics_coords, terms_probs, top_5_doc = btm_analysis(
                            combined_dataset, dic_path
                        )
                    else:
                        topics_coords, terms_probs, top_5_doc = btm_analysis(
                            combined_dataset
                        )

                    topics_coords.to_csv(path1, index=False)
                    for topic, df in terms_probs.items():
                        path2 = os.path.join(
                            save_path3, f"{formatted_date}_{topic}.csv"
                        )
                        df.to_csv(path2, index=False)
                    for topic, df in top_5_doc.items():
                        path3 = os.path.join(
                            save_path4, f"{formatted_date}_{topic}.csv"
                        )
                        df.to_csv(path3, index=False)

                # 下載立場原始資料
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
                    if "prediction_filtered" in raw_predict_data.columns:
                        raw_predict_data.drop(
                            columns="prediction_filtered", inplace=True
                        )
                        raw_predict_data.rename(
                            columns={"prediction_raw": "prediction"}, inplace=True
                        )

    if startDate1 and endDate1:
        startDate1 = datetime.strptime(startDate1, "%Y-%m-%d")
        startDate1 = startDate1.strftime("%Y%m%d")
        endDate1 = datetime.strptime(endDate1, "%Y-%m-%d")
        endDate1 = endDate1.strftime("%Y%m%d")

        data = pd.read_csv(
            os.path.join(full_path, f"事件一：{startDate1}_{endDate1}.csv")
        )
        path = os.path.join(save_path, f"事件一：{startDate1}_{endDate1}.csv")
        degree_path = os.path.join(
            save_path5, f"事件一：{startDate1}_{endDate1}_degree.json"
        )
        betweeness_path = os.path.join(
            save_path5, f"事件一：{startDate1}_{endDate1}_betweeness.json"
        )
        closeness_path = os.path.join(
            save_path5, f"事件一：{startDate1}_{endDate1}_closeness.json"
        )
        eigenvector_path = os.path.join(
            save_path5, f"事件一：{startDate1}_{endDate1}_eigenvector.json"
        )

        (
            event_scores1,
            event1_network_degree,
            event1_network_betweeness,
            event1_network_closeness,
            event1_network_eigenvector,
            result,
            filtered_dataset,
        ) = centerityScore(data, path)
        # event_scores1 = centerityScore(data,path)

        if not os.path.exists(path):
            event_scores1.to_csv(path, index=False)
        with open(degree_path, "w") as file:
            json.dump(event1_network_degree, file, indent=4)
        with open(betweeness_path, "w") as file:
            json.dump(event1_network_betweeness, file, indent=4)
        with open(closeness_path, "w") as file:
            json.dump(event1_network_closeness, file, indent=4)
        with open(eigenvector_path, "w") as file:
            json.dump(event1_network_eigenvector, file, indent=4)
        # 不同時間點立場
        temp_df = pd.DataFrame(
            {
                "user": result["key"],
                "name": result["label"],
                f"事件一：{startDate1}_{endDate1}": result["cluster"],
            }
        )
        if Stance.empty:
            Stance = temp_df
        else:
            Stance = pd.merge(Stance, temp_df, on=["user", "name"], how="outer")
        # btm
        path1 = os.path.join(save_path2, f"事件一：{startDate1}_{endDate1}.csv")
        if not os.path.exists(path1):
            if dic_path:
                topics_coords, terms_probs, top_5_doc = btm_analysis(data, dic_path)
            else:
                topics_coords, terms_probs, top_5_doc = btm_analysis(data)
            topics_coords.to_csv(path1, index=False)
            for topic, df in terms_probs.items():
                path2 = os.path.join(
                    save_path3, f"事件一：{startDate1}_{endDate1}_{topic}.csv"
                )
                df.to_csv(path2, index=False)
            for topic, df in top_5_doc.items():
                path3 = os.path.join(
                    save_path4, f"事件一：{startDate1}_{endDate1}_{topic}.csv"
                )
                df.to_csv(path3, index=False)
        # 下載立場原始資料
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
            if "prediction_filtered" in raw_predict_data.columns:
                raw_predict_data.drop(columns="prediction_filtered", inplace=True)
                raw_predict_data.rename(
                    columns={"prediction_raw": "prediction"}, inplace=True
                )

    if startDate2 and endDate2:
        startDate2 = datetime.strptime(startDate2, "%Y-%m-%d")
        startDate2 = startDate2.strftime("%Y%m%d")
        endDate2 = datetime.strptime(endDate2, "%Y-%m-%d")
        endDate2 = endDate2.strftime("%Y%m%d")
        data = pd.read_csv(
            os.path.join(full_path, f"事件二：{startDate2}_{endDate2}.csv")
        )

        path = os.path.join(save_path, f"事件二：{startDate2}_{endDate2}.csv")
        degree_path = os.path.join(
            save_path5, f"事件二：{startDate2}_{endDate2}_degree.json"
        )
        betweeness_path = os.path.join(
            save_path5, f"事件二：{startDate2}_{endDate2}_betweeness.json"
        )
        closeness_path = os.path.join(
            save_path5, f"事件二：{startDate2}_{endDate2}_closeness.json"
        )
        eigenvector_path = os.path.join(
            save_path5, f"事件二：{startDate2}_{endDate2}_eigenvector.json"
        )

        (
            event_scores2,
            event2_network_degree,
            event2_network_betweeness,
            event2_network_closeness,
            event2_network_eigenvector,
            result,
            filtered_dataset,
        ) = centerityScore(data, path)
        # event_scores2 = centerityScore(data,path)

        if not os.path.exists(path):
            event_scores2.to_csv(path, index=False)
        with open(degree_path, "w") as file:
            json.dump(event2_network_degree, file, indent=4)
        with open(betweeness_path, "w") as file:
            json.dump(event2_network_betweeness, file, indent=4)
        with open(closeness_path, "w") as file:
            json.dump(event2_network_closeness, file, indent=4)
        with open(eigenvector_path, "w") as file:
            json.dump(event2_network_eigenvector, file, indent=4)
        # 不同時間點立場
        temp_df = pd.DataFrame(
            {
                "user": result["key"],
                "name": result["label"],
                f"{formatted_date}": result["cluster"],
            }
        )
        if Stance.empty:
            Stance = temp_df
        else:
            Stance = pd.merge(Stance, temp_df, on=["user", "name"], how="outer")
        # btm
        path1 = os.path.join(save_path2, f"事件二：{startDate2}_{endDate2}.csv")
        if not os.path.exists(path1):
            if dic_path:
                topics_coords, terms_probs, top_5_doc = btm_analysis(data, dic_path)
            else:
                topics_coords, terms_probs, top_5_doc = btm_analysis(data)
            topics_coords.to_csv(path1, index=False)
            for topic, df in terms_probs.items():
                path2 = os.path.join(
                    save_path3, f"事件二：{startDate1}_{endDate1}_{topic}.csv"
                )
                df.to_csv(path2, index=False)
            for topic, df in top_5_doc.items():
                path3 = os.path.join(
                    save_path4, f"事件二：{startDate1}_{endDate1}_{topic}.csv"
                )
                df.to_csv(path3, index=False)
        # 下載立場原始資料
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
            if "prediction_filtered" in raw_predict_data.columns:
                raw_predict_data.drop(columns="prediction_filtered", inplace=True)
                raw_predict_data.rename(
                    columns={"prediction_raw": "prediction"}, inplace=True
                )

    if startDate3 and endDate3:
        startDate3 = datetime.strptime(startDate3, "%Y-%m-%d")
        startDate3 = startDate3.strftime("%Y%m%d")
        endDate3 = datetime.strptime(endDate3, "%Y-%m-%d")
        endDate3 = endDate3.strftime("%Y%m%d")
        data = pd.read_csv(
            os.path.join(full_path, f"事件三：{startDate3}_{endDate3}.csv")
        )

        path = os.path.join(save_path, f"事件三：{startDate3}_{endDate3}..csv")
        degree_path = os.path.join(
            save_path5, f"事件三：{startDate3}_{endDate3}_degree.json"
        )
        betweeness_path = os.path.join(
            save_path5, f"事件三：{startDate3}_{endDate3}_betweeness.json"
        )
        closeness_path = os.path.join(
            save_path5, f"事件三：{startDate3}_{endDate3}_closeness.json"
        )
        eigenvector_path = os.path.join(
            save_path5, f"事件三：{startDate3}_{endDate3}_eigenvector.json"
        )

        (
            event_scores3,
            event3_network_degree,
            event3_network_betweeness,
            event3_network_closeness,
            event3_network_eigenvector,
            result,
            filtered_dataset,
        ) = centerityScore(data, path)
        # event_scores3 = centerityScore(data,path)

        if not os.path.exists(path):
            event_scores3.to_csv(path, index=False)
        with open(degree_path, "w") as file:
            json.dump(event3_network_degree, file, indent=4)
        with open(betweeness_path, "w") as file:
            json.dump(event3_network_betweeness, file, indent=4)
        with open(closeness_path, "w") as file:
            json.dump(event3_network_closeness, file, indent=4)
        with open(eigenvector_path, "w") as file:
            json.dump(event3_network_eigenvector, file, indent=4)
        # 不同時間點立場
        temp_df = pd.DataFrame(
            {
                "user": result["key"],
                "name": result["label"],
                f"{formatted_date}": result["cluster"],
            }
        )
        if Stance.empty:
            Stance = temp_df
        else:
            Stance = pd.merge(Stance, temp_df, on=["user", "name"], how="outer")
        # btm
        path1 = os.path.join(save_path2, f"事件三：{startDate3}_{endDate3}.csv")
        if not os.path.exists(path1):
            if dic_path:
                topics_coords, terms_probs, top_5_doc = btm_analysis(data, dic_path)
            else:
                topics_coords, terms_probs, top_5_doc = btm_analysis(data)
            topics_coords.to_csv(path1, index=False)
            for topic, df in terms_probs.items():
                path2 = os.path.join(
                    save_path3, f"事件三：{startDate1}_{endDate1}_{topic}.csv"
                )
                df.to_csv(path2, index=False)
            for topic, df in top_5_doc.items():
                path3 = os.path.join(
                    save_path4, f"事件三：{startDate1}_{endDate1}_{topic}.csv"
                )
                df.to_csv(path3, index=False)
        # 下載立場原始資料
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
            if "prediction_filtered" in raw_predict_data.columns:
                raw_predict_data.drop(columns="prediction_filtered", inplace=True)
                raw_predict_data.rename(
                    columns={"prediction_raw": "prediction"}, inplace=True
                )

    Stance.fillna("無資料", inplace=True)
    sorted_columns = sorted(Stance.columns, key=lambda x: (x.isdigit(), x))
    Stance = Stance[sorted_columns]

    stancefolder = "stance"
    stancepath = os.path.join(save_folder, stancefolder)
    if not os.path.exists(stancepath):
        os.makedirs(stancepath)
    Stance.to_csv(os.path.join(stancepath, "Stance.csv"), index=False)

    downloadfolder = "download"
    downloadpath = os.path.join(save_folder, downloadfolder)
    if not os.path.exists(downloadpath):
        os.makedirs(downloadpath)
    raw_predict_data.to_csv(os.path.join(downloadpath, "Download.csv"), index=False)

    return jsonify({"result": "計算成功"})


@app.route("/download", methods=["POST"])
def download():
    filename = request.form.get("filename")
    print(filename)
    folder = f"{filename}"
    folder2 = "download"
    full_path = os.path.join(folder, folder2)
    file = pd.read_csv(os.path.join(full_path, "Download.csv"))
    csv = file.to_csv(index=False)

    return csv


@app.route("/get_file", methods=["POST"])
def get_file():
    csv_files = glob.glob("data/*.csv")
    titles = [os.path.splitext(os.path.basename(f))[0] for f in csv_files]
    return jsonify(titles)


@app.route("/result1", methods=["POST"])
def result1():
    filename = request.form.get("filename")  # 獲取 filename
    filename = f"{filename}.csv"
    data_folder = "data"
    full_path = os.path.join(data_folder, filename)
    # print(full_path)
    if not os.path.exists(full_path):
        return jsonify({"error": "檔案不存在"})

    data = pd.read_csv(full_path)  # 讀取檔案
    result = statisticCalcu(data)
    # print(result)
    result = result.to_json()
    return jsonify({"result": result})


@app.route("/chartData", methods=["POST"])
def chartData():
    filename = request.form.get("filename")  # 獲取 filename
    filename = f"{filename}.csv"
    data_folder = "data"
    full_path = os.path.join(data_folder, filename)
    # print(full_path)
    if not os.path.exists(full_path):
        return jsonify({"error": "檔案不存在"})
    data = pd.read_csv(full_path)  # 讀取檔案
    data["created_at"] = pd.to_datetime(data["created_at"])
    data = data.groupby(data["created_at"].dt.date).size()
    chart_data = {
        "labels": [date.strftime("%Y-%m-%d") for date in data.index],
        "datasets": [
            {
                "label": "Number of tweets",
                "data": data.values.tolist(),
                "borderColor": "rgb(255, 99, 132)",
                "backgroundColor": "rgba(255, 99, 132, 0.5)",
            }
        ],
    }

    return jsonify(chart_data)


@app.route("/result2", methods=["POST"])
def result2():
    filename = request.form.get("filename")  # 獲取 filename
    folder = f"{filename}"
    folder2 = "centerity"
    full_path = os.path.join(folder, folder2)

    files = os.listdir(full_path)  # 獲取目錄下的所有文件

    result = {}
    for csv_file in files:
        data = pd.read_csv(os.path.join(full_path, csv_file))  # 讀取 CSV 文件
        result[csv_file] = data.to_dict()  # 將 DataFrame 轉換為字典並存儲在結果中
    return jsonify({"result": result})


@app.route("/get_cendata", methods=["POST"])
def get_cendata():
    filename = request.form.get("filename")  # 獲取 filename
    # print(filename)
    folder = f"{filename}"
    folder2 = "centerity"
    full_path = os.path.join(folder, folder2)
    files = os.listdir(full_path)  # 獲取目錄下的所有文件
    item = request.form.get("selectedItem")
    # print(item)
    data = []
    for file in files:
        # if file.endswith('.csv'):
        df = pd.read_csv(os.path.join(full_path, file))
        values = df[df["Account"] == item]["betweenness_centrality_Score"].values
        if len(values) > 0:
            centrality = values[0]
        else:
            centrality = 0
        data.append({"name": file.split(".")[0], "centrality": centrality})
    # print(data)
    return jsonify({"cenData": data})


@app.route("/btm_fig", methods=["POST"])
def btm_fig():
    filename = request.form.get("filename")  # 獲取 filename
    folder = f"{filename}"
    folder2 = "btm"
    folder3 = "topics_coords"
    full_path = os.path.join(folder, folder2, folder3)

    files = os.listdir(full_path)  # 獲取目錄下的所有文件

    btm_fig = {}
    for csv_file in files:
        data = pd.read_csv(os.path.join(full_path, csv_file))  # 讀取 CSV 文件
        btm_fig[csv_file] = data.to_dict()  # 將 DataFrame 轉換為字典並存儲在結果中
    return jsonify({"btm_fig": btm_fig})


@app.route("/btm_terms", methods=["POST"])
def btm_terms():
    filename = request.form.get("filename")  # 獲取 filename
    folder = f"{filename}"
    folder2 = "btm"
    folder3 = "terms_probs"
    full_path = os.path.join(folder, folder2, folder3)
    topic = request.form.get("selectedBubble")
    # print(topic)

    files = os.listdir(full_path)  # 獲取目錄下的所有文件

    btm_terms = {}
    for csv_file in files:
        if csv_file.endswith(f"{topic}.csv"):
            # print(csv_file)
            data = pd.read_csv(os.path.join(full_path, csv_file))  # 讀取 CSV 文件
            btm_terms[csv_file] = (
                data.to_dict()
            )  # 將 DataFrame 轉換為字典並存儲在結果中
    # btm_terms = data.to_json()
    return jsonify({"btm_terms": btm_terms})


@app.route("/btm_doc", methods=["POST"])
def btm_doc():
    filename = request.form.get("filename")  # 獲取 filename
    folder = f"{filename}"
    folder2 = "btm"
    folder3 = "top_5_doc"
    full_path = os.path.join(folder, folder2, folder3)
    topic = request.form.get("selectedBubble")
    files = os.listdir(full_path)  # 獲取目錄下的所有文件

    top_5_doc = {}
    for csv_file in files:
        if csv_file.endswith(f"{topic}.csv"):
            data = pd.read_csv(os.path.join(full_path, csv_file))  # 讀取 CSV 文件
            top_5_doc[csv_file] = (
                data.to_dict()
            )  # 將 DataFrame 轉換為字典並存儲在結果中
    # btm_terms = data.to_json()
    return jsonify({"top_5_doc": top_5_doc})


@app.route("/network", methods=["POST"])
def network_route():
    filename = request.form.get("filename")  # 獲取 filename
    folder = f"{filename}"
    folder2 = "network"
    full_path = os.path.join(folder, folder2)
    cen_type = request.form.get("cen_type")
    print(cen_type)
    range = request.form.get("selectedFile")
    print(range)
    files = os.listdir(full_path)
    # files = os.listdir(full_path)  # 獲取目錄下的所有文件

    # network = {}
    for json_file in files:
        if json_file.endswith(f"{range}_{cen_type}.json"):
            with open(os.path.join(full_path, json_file)) as f:
                data = json.load(f)
            # data = pd.read_json(os.path.join(full_path, json_file))
            # network[json_file] = data.to_dict()  # 將 DataFrame 轉換為字典並存儲在結果中
    # btm_terms = data.to_json()
    return jsonify(data)


@app.route("/stance", methods=["POST"])
def stance():
    filename = request.form.get("filename")
    selected_col = request.form.get("selectedFile")
    folder = f"{filename}"
    folder2 = "stance"
    full_path = os.path.join(folder, folder2)
    data = pd.read_csv(os.path.join(full_path, "Stance.csv"))

    compare = data.drop("user", axis=1)
    if selected_col not in data.columns:
        return "Selected column not found in the data."
    col_index = compare.columns.get_loc(selected_col)
    if col_index == 1:
        return jsonify(
            {
                "result": "Selected column is the first column, no previous column available."
            }
        )
    previous_col = compare.columns[col_index - 1]
    filtered_compare = compare[
        (compare[selected_col] != "無資料") & (compare[previous_col] != "無資料")
    ]
    filtered_compare = filtered_compare[
        (filtered_compare[selected_col] != "無貼文判斷立場")
        & (filtered_compare[previous_col] != "無貼文判斷立場")
    ]
    changes = filtered_compare[
        filtered_compare[selected_col] != filtered_compare[previous_col]
    ]
    changes = changes.merge(data[["user"]], left_index=True, right_index=True)
    results = changes.apply(
        lambda row: [row["user"], row[previous_col], row[selected_col]], axis=1
    ).tolist()
    # results = changes.apply(
    # lambda row: {
    #     'user': row['user'],  # 假设 'user' 是识别用户的列
    #     'original_position': row[previous_col],
    #     'new_position': row[selected_col]
    # },
    # axis=1
    # ).tolist()
    return jsonify(results)


@app.route("/check-file", methods=["POST"])
def check_file():
    filename = request.form.get("filename")
    folder = f"{filename}"
    folder2 = "stance"
    full_path = os.path.join(folder, folder2)
    full_path = os.path.join(full_path, "Stance.csv")
    print(full_path)
    file_exists = os.path.exists(full_path)
    print(file_exists)
    return jsonify({"exists": file_exists})


@app.route("/stanceDetail", methods=["POST"])
def stanceDetail():
    filename = request.form.get("filename")
    time = request.form.get("time")
    user = request.form.get("user")
    folder = f"{filename}.csv"
    folder2 = "data"
    full_path = os.path.join(folder2, folder)
    #     print(full_path)
    data = pd.read_csv(full_path)
    data = data[data["from_user_name"] == user]
    data["created_at"] = pd.to_datetime(data["created_at"])

    time_datetime = datetime.strptime(time, "%Y%m%d")
    new_start = time_datetime - timedelta(weeks=2) + timedelta(days=1)
    new_end = time_datetime + timedelta(days=1)
    old_start = time_datetime - timedelta(weeks=3) + timedelta(days=1)
    old_end = time_datetime - timedelta(weeks=1) + timedelta(days=1)

    olddata = data[(data["created_at"] >= old_start) & (data["created_at"] <= old_end)]
    newdata = data[(data["created_at"] >= new_start) & (data["created_at"] <= new_end)]
    selec = ["created_at", "from_user_name", "from_user_realname", "text"]
    olddata = olddata[selec]
    newdata = newdata[selec]

    olddata["created_at"] = olddata["created_at"].apply(
        lambda x: x.strftime("%Y-%m-%d %H:%M")
    )
    newdata["created_at"] = newdata["created_at"].apply(
        lambda x: x.strftime("%Y-%m-%d %H:%M")
    )

    data = {"old": olddata.to_dict(), "new": newdata.to_dict()}
    #     print(data)

    return jsonify({"data": data})


@app.route("/timeline", methods=["POST"])
def timeline():
    filename = request.form.get("filename")  # 獲取 filename
    file_name = f"{filename}.csv"
    # print(filename)

    folder = f"{filename}"
    data_folder = "data"
    full_path = os.path.join(data_folder, file_name)
    data = pd.read_csv(full_path)
    # 分兩週兩週算中心性分數
    # centerity_scores = centerityScore(data)
    data["created_at"] = pd.to_datetime(data["created_at"])
    data["period"] = data["created_at"].dt.to_period("W-SUN")
    groups = data.groupby("period")
    datasets = datasets = {}
    for period, group in groups:
        datasets[period] = group
    datasets_list = list(datasets.values())
    highlightPeriods = []
    for i in range(len(datasets_list) - 1):
        combined_dataset = pd.concat([datasets_list[i], datasets_list[i + 1]])
        period = {}
        start = combined_dataset["period"].min().start_time.strftime("%Y%m%d")
        end = combined_dataset["period"].max().end_time.strftime("%Y%m%d")
        period["fileName"] = combined_dataset["created_at"].max().strftime("%Y%m%d")
        period["startDate"] = start
        period["endDate"] = end
        highlightPeriods.append(period)

    folder2 = "centerity"
    full_path = os.path.join(folder, folder2)
    files = os.listdir(full_path)
    highlightPeriods2 = []
    for csv_file in files:
        if csv_file.startswith("事"):
            period = {}
            file = csv_file.replace(".csv", "")
            date = file.split("：")[1]
            start = date.split("_")[0]
            end = date.split("_")[1]
            period["fileName"] = file
            period["startDate"] = start
            period["endDate"] = end
            highlightPeriods2.append(period)
    first = pd.to_datetime(min(p["startDate"] for p in highlightPeriods)) - timedelta(
        days=1
    )
    first = first.strftime("%Y%m%d")
    last = pd.to_datetime(max(p["endDate"] for p in highlightPeriods)) + timedelta(
        days=1
    )
    last = last.strftime("%Y%m%d")
    print(highlightPeriods2)
    return jsonify(
        {
            "highlightPeriods": highlightPeriods,
            "highlightPeriods2": highlightPeriods2,
            "first": first,
            "last": last,
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
