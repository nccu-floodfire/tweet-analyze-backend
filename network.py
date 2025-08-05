import json

import networkx as nx
import numpy as np
import pandas as pd
import torch
from node2vec import Node2Vec
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import BertForSequenceClassification, BertTokenizer


def network(data, data2):
    # 創建一個圖
    edges = data.values.tolist()
    G = nx.Graph()
    for from_user, rt_user, count in edges:
        G.add_edge(from_user, rt_user, weight=count)
    # 計算中心性
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G, normalized=True)
    closeness_centrality = nx.closeness_centrality(G)
    eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
    centrality_df = pd.DataFrame(
        {
            "from_user_name": list(degree_centrality.keys()),
            "Degree Centrality": list(degree_centrality.values()),
            "Betweenness Centrality": list(betweenness_centrality.values()),
            "Closeness Centrality": list(closeness_centrality.values()),
            "Eigenvector Centrality": list(eigenvector_centrality.values()),
        }
    )
    # 生成節點向量位置
    node2vec = Node2Vec(
        G, dimensions=128, walk_length=10, num_walks=10, p=1, q=1, workers=4
    )  # p, q是控制隨機遊走的參數
    model = node2vec.fit(window=10, min_count=1, sg=1)  # 使用 Skip-gram 模型
    embeddings = {node: model.wv[node] for node in model.wv.index_to_key}
    embed_matrix = np.array(list(embeddings.values()))
    n_samples = embed_matrix.shape[0]
    perplexity_value = min(30, n_samples - 1)  # 确保不超过样本数减一
    tsne = TSNE(
        n_components=2, learning_rate="auto", init="random", perplexity=perplexity_value
    )
    embed_2d_tsne = tsne.fit_transform(embed_matrix)
    users = list(embeddings.keys())
    user_embeddings_2d = np.column_stack((users, embed_2d_tsne))
    df_embeddings = pd.DataFrame(
        user_embeddings_2d, columns=["from_user_name", "x", "y"]
    )
    # 製作立場分類需要的資料集
    user = df_embeddings["from_user_name"].tolist()
    filtered_dataset = data2[data2["from_user_name"].isin(user)]
    # 結果圖1（有from_user_name, x, y, Degree Centrality, Betweenness Centrality, Closeness Centrality, Eigenvector Centrality）
    result = df_embeddings.merge(centrality_df, on="from_user_name", how="left")

    # 立場分類模型
    new_texts = filtered_dataset["text"].tolist()
    batch_size = 16
    tokenizer = BertTokenizer.from_pretrained("./saved_model")
    encodings = tokenizer(
        new_texts, truncation=True, padding=True, max_length=512, return_tensors="pt"
    )
    dataset = TensorDataset(encodings["input_ids"], encodings["attention_mask"])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model = BertForSequenceClassification.from_pretrained("./saved_model")
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Processing batches"):
            input_ids, attention_mask = batch
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            batch_predictions = logits.argmax(dim=-1)
            predictions.extend(batch_predictions.tolist())

    type_mapping = {"A": 0, "B": 1, "C": 2, "F": 3, "G1": 4, "G2": 5, "H": 6, "J": 7}
    reverse_mapping = {v: k for k, v in type_mapping.items()}
    predicted_labels = [reverse_mapping[label_id] for label_id in predictions]
    df = pd.DataFrame({"text": new_texts, "prediction": predicted_labels})
    filtered_dataset = filtered_dataset.merge(df, on="text")
    # 根據貼文預測結果為每個用戶分配立場
    priority_order = ["J", "H", "G2", "B", "G1", "C", "F", "A"]
    grouped = (
        filtered_dataset.groupby("from_user_name")["prediction"]
        .value_counts()
        .unstack(fill_value=0)
    )
    for category in priority_order:
        if category not in grouped.columns:
            grouped[category] = 0  # 添加缺失的类别列，用 0 填充
    grouped = grouped[priority_order]

    def select_prediction(row):
        # 找到值最大的列的索引，如果有多个最大值，返回第一个（按优先顺序）
        return row.idxmax()

    user_stance = grouped.apply(select_prediction, axis=1)
    stance = user_stance.to_frame(name="Stance")
    stance = stance.reset_index()
    # 結果圖2（有from_user_name, Stance, x, y, centerity...）
    result = result.merge(stance, on="from_user_name", how="outer")
    # 加入用戶真實名稱
    select = ["from_user_name", "from_user_realname"]
    realname = filtered_dataset[select]
    realname = realname.drop_duplicates(subset="from_user_name", keep="last")
    # 結果圖3（有from_user_name, from_user_realname, Stance, x, y, centerity...）
    result = result.merge(realname, on="from_user_name", how="left")
    # 加入用戶類型
    path = "./0428Usertype.xlsx"
    usertype = pd.read_excel(path, sheet_name="用戶能見度")
    usertype = usertype[
        (usertype["類型"] == "新聞媒體") | (usertype["類型"] == "名人或意見領袖")
    ]
    select = ["帳號 @user", "類型"]
    usertype = usertype[select]
    usertype.columns = ["from_user_name", "type"]
    result = result.merge(usertype, on="from_user_name", how="left")
    result["Stance"] = result["Stance"].fillna("無貼文判斷立場")
    result["from_user_realname"] = result["from_user_realname"].fillna("無資料")
    result["type"] = result["type"].fillna("一般帳號")
    # 轉成網路圖的格式
    result = result.rename(
        columns={
            "from_user_name": "key",
            "from_user_realname": "label",
            "Stance": "cluster",
            "type": "tag",
            "x": "x",
            "y": "y",
        }
    )

    edges = [[edge[0], edge[1]] for edge in edges]
    degree_node = [
        {
            "key": row["key"],
            "label": row["label"],
            "tag": row["tag"],
            "cluster": row["cluster"],
            "x": row["x"],
            "y": row["y"],
            "score": row["Degree Centrality"],
        }
        for index, row in result.iterrows()
    ]

    betweenness_node = [
        {
            "key": row["key"],
            "label": row["label"],
            "tag": row["tag"],
            "cluster": row["cluster"],
            "x": row["x"],
            "y": row["y"],
            "score": row["Betweenness Centrality"],
        }
        for index, row in result.iterrows()
    ]

    closeness_node = [
        {
            "key": row["key"],
            "label": row["label"],
            "tag": row["tag"],
            "cluster": row["cluster"],
            "x": row["x"],
            "y": row["y"],
            "score": row["Closeness Centrality"],
        }
        for index, row in result.iterrows()
    ]

    eigenvector_node = [
        {
            "key": row["key"],
            "label": row["label"],
            "tag": row["tag"],
            "cluster": row["cluster"],
            "x": row["x"],
            "y": row["y"],
            "score": row["Eigenvector Centrality"],
        }
        for index, row in result.iterrows()
    ]

    cluster = [
        {"key": "A", "color": "#FEF4DC", "clusterLabel": "中立"},
        {"key": "B", "color": "#D85656", "clusterLabel": "支持中國共產黨及習近平"},
        {"key": "C", "color": "#F7826C", "clusterLabel": "支持中國共產黨"},
        {"key": "F", "color": "#B2D0E3", "clusterLabel": "反對中國"},
        {"key": "G1", "color": "#95AAA2", "clusterLabel": "反對中國及習近平"},
        {
            "key": "G2",
            "color": "#B5E3EA",
            "clusterLabel": "反對中國及習近平但支持共產黨",
        },
        {
            "key": "H",
            "color": "#C3D4C8",
            "clusterLabel": "反對中國及共產黨但支持習近平",
        },
        {"key": "J", "color": "#5776B5", "clusterLabel": "反對中國共產黨及習近平"},
        {"key": "無貼文判斷立場", "color": "#F5F5F2", "clusterLabel": "無貼文判斷立場"},
    ]

    tag = [
        {"key": "一般帳號", "image": "person.svg"},
        {"key": "新聞媒體", "image": "company.svg"},
        {"key": "名人或意見領袖", "image": "field.svg"},
    ]
    output_degree = {
        "nodes": degree_node,
        "edges": edges,
        "clusters": cluster,
        "tags": tag,
    }
    output_betweenness = {
        "nodes": betweenness_node,
        "edges": edges,
        "clusters": cluster,
        "tags": tag,
    }
    output_closeness = {
        "nodes": closeness_node,
        "edges": edges,
        "clusters": cluster,
        "tags": tag,
    }
    output_eigenvector = {
        "nodes": eigenvector_node,
        "edges": edges,
        "clusters": cluster,
        "tags": tag,
    }

    return (
        output_degree,
        output_betweenness,
        output_closeness,
        output_eigenvector,
        result,
        filtered_dataset,
    )
