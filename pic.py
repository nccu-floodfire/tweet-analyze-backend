import json
import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image

# 讀取 JSON 文件
with open('/Users/changwen/Desktop/論文系統/TweetAnalyze/backend/data.csv_2022-10-14_2022-10-24_2022-11-24_2022-12-09_None_None/network/事件一：20221014_20221024_betweeness.json', 'r') as f:
    data = json.load(f)

nodes = data['nodes']
edges = data['edges']

# 建立網絡圖
G = nx.Graph()

# 加入節點
for node in nodes:
    G.add_node(node['key'], label=node['label'], tag=node['tag'], cluster=node['cluster'], pos=(float(node['x']), float(node['y'])))
    # 将 x 和 y 转换为浮点数


# 加入邊
G.add_edges_from(edges)

# 繪製網絡圖
pos = nx.get_node_attributes(G, 'pos')
labels = nx.get_node_attributes(G, 'label')

plt.figure(figsize=(10, 8))
nx.draw(G, pos, with_labels=True, labels=labels, node_size=500, node_color="lightblue", font_size=10, font_color="black")

# 儲存為 JPG 檔案
plt.savefig("network_graph.jpg", format="jpg")
plt.show()

# 開啟並顯示 JPG 檔案
img = Image.open("network_graph.jpg")
img.show()
