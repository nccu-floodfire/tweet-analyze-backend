import json
import os
import re

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from network import network


def centralityScore(data, path):
    # data為兩週兩週的資料
    original = data
    data = original[original["retweet_count"] > 1]
    data = data[
        (data["lang"] == "en") | (data["lang"] == "zh") | (data["lang"] == "ja")
    ]
    # 新增RT_users欄
    pattern = r"RT @\w+"
    data.loc[:, "rt_users"] = data["text"].apply(lambda x: re.findall(pattern, x))
    data.loc[:, "rt_users"] = data["rt_users"].apply(lambda x: x[0] if x else "")
    data.loc[:, "rt_users"] = data["rt_users"].apply(lambda x: re.sub("RT @", "", x))
    # 製作追蹤者表
    follower = ["from_user_name", "from_user_realname", "from_user_followercount"]
    follower = data[follower]
    follower.columns = ["Account", "Name", "Follower"]
    follower = follower.groupby(["Account", "Name"]).last().reset_index()
    # 過濾掉不需要的欄位
    filterer = ["from_user_name", "rt_users"]
    data = data[filterer]
    # 刪掉有空值的組合
    data = data[(data["from_user_name"] != "") & (data["rt_users"] != "")]
    # 計算每個轉推組合次數
    data = data.groupby(["from_user_name", "rt_users"]).size().reset_index(name="count")

    # 要生成社群網路圖的資料的轉推次數要大於5
    network_data = data[data["count"] > 5]
    (
        output_degree,
        output_betweenness,
        output_closeness,
        output_eigenvector,
        result,
        filtered_dataset,
    ) = network(network_data, original)

    if not os.path.exists(path):
        # 生成有向圖算中心性分數
        G = nx.from_pandas_edgelist(
            data, "from_user_name", "rt_users", "count", create_using=nx.DiGraph()
        )
        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G)
        closeness_centrality = nx.closeness_centrality(G)
        eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=500)
        df_degree_centrality = pd.DataFrame(
            list(degree_centrality.items()),
            columns=["Account", "degree_centrality_Score"],
        )
        df_betweenness_centrality = pd.DataFrame(
            list(betweenness_centrality.items()),
            columns=["Account", "betweenness_centrality_Score"],
        )
        df_closeness_centrality = pd.DataFrame(
            list(closeness_centrality.items()),
            columns=["Account", "closeness_centrality_Score"],
        )
        df_eigenvector_centrality = pd.DataFrame(
            list(eigenvector_centrality.items()),
            columns=["Account", "eigenvector_centrality_Score"],
        )
        df_betweenness_centrality = df_betweenness_centrality[
            "betweenness_centrality_Score"
        ]
        df_closeness_centrality = df_closeness_centrality["closeness_centrality_Score"]
        df_eigenvector_centrality = df_eigenvector_centrality[
            "eigenvector_centrality_Score"
        ]
        centrality_score = pd.concat(
            [
                df_degree_centrality,
                df_betweenness_centrality,
                df_closeness_centrality,
                df_eigenvector_centrality,
            ],
            axis=1,
        )
        # 加入追蹤人數資訊
        table = pd.merge(centrality_score, follower, on="Account", how="inner")
        # 移除不重要節點
        table = table[
            ~(
                (table["betweenness_centrality_Score"] == 0)
                & (table["closeness_centrality_Score"] == 0)
            )
        ]
    else:
        table = pd.DataFrame()

    # return table
    return (
        table,
        output_degree,
        output_betweenness,
        output_closeness,
        output_eigenvector,
        result,
        filtered_dataset,
    )
