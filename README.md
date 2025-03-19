# Tweet Data Visualization Tool (後端）

### 啟動後端專案

```
python server.py
```

### 專案介紹

- 此Repo為**Tweet Data Visualization Tool**的後端專案
- 以Python進行開發

#### 主要功能分成3塊

- 首頁功能`upload`及`centerity`，分別為首頁的上傳功能及分析中心性資料並儲存
<img width="329" alt="截圖 2025-03-19 晚上9 59 23" src="https://github.com/user-attachments/assets/9b354593-b4de-4fa1-a399-42de27f3af8c" />

- 資料列表頁功能`get_file`及`check-file`，分別為列出有哪些資料已上傳，以及在點擊資料欲查看分析時，判斷資料是否已分析完畢
<img width="321" alt="截圖 2025-03-19 晚上9 59 31" src="https://github.com/user-attachments/assets/60e8efdc-8db9-48d9-8e73-b3f210cf6fc7" />

- 分析結果頁功能
  - `result1`及`chartData`:資料概況功能，計算表格內容及趨勢圖所需資訊
  - `result2`及`get_cendata`:社群網路中心性分數功能，計算表格內容及趨勢圖所需資訊
  - `btm_fig`、`btm_terms`及`btm_doc`:主題分類功能，計算圓餅圖、趨勢圖、表格所需資訊
  - `network`、`download`及`stance`:社群網路圖功能，計算網路圖所需資料、下載功能及切換時間時立場變化之資料
  - `timeline`:時間軸功能，算出資料及所在時間範圍及區間
  - `stanceDetail`:立場變化頁面功能，找出節點立場前後變化之貼文內容
<img width="288" alt="截圖 2025-03-19 晚上10 00 01" src="https://github.com/user-attachments/assets/1874b281-4a12-42eb-8d68-610c5b48de44" />

- 注意：因立場分類所使用之Bert模型在開發時為載模型到本機使用，此模型無法上傳github，需手動下載並加入專案後，才可正常執行，位置放在`saved_model`資料夾下，[模型下載連結](https://drive.google.com/file/d/1Wkmv7rGtmyrjt5XIId_d2fus98L27hJj/view?usp=sharing)
<img width="1084" alt="截圖 2025-03-19 晚上9 53 35" src="https://github.com/user-attachments/assets/520d840b-7912-417b-9a30-c292a7247a52" />

