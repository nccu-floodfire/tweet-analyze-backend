#!/usr/bin/env python
import os
import sys

import pandas as pd

# 1. 檢查命令列參數
if len(sys.argv) < 2:
    print("使用方式：python your_script_name.py <輸入檔案路徑>")
    print("範例：python your_script_name.py data.csv")
    sys.exit(1)  # 如果沒有提供檔案路徑，則退出程式

# 從命令列參數取得輸入檔案路徑
input_filepath = sys.argv[1]

# 2. 根據輸入檔案路徑生成輸出檔案名稱
# 獲取檔案所在的目錄路徑
directory = os.path.dirname(input_filepath)
# 獲取不含路徑的檔案名稱和副檔名
filename_without_ext, file_extension = os.path.splitext(
    os.path.basename(input_filepath)
)

# 組合成新的輸出檔案名稱
output_filename = f"{filename_without_ext}_sorted{file_extension}"
# 組合完整的輸出檔案路徑
output_filepath = os.path.join(directory, output_filename)

# 3. 載入 CSV 檔案
try:
    df = pd.read_csv(input_filepath)
    print(f"成功載入檔案：{input_filepath}")
except FileNotFoundError:
    print(f"錯誤：找不到檔案 '{input_filepath}'。請確認檔案路徑是否正確。")
    sys.exit(1)
except Exception as e:
    print(f"讀取檔案時發生錯誤：{e}")
    sys.exit(1)

# 4. 定義您想要的**完整**欄位順序
# 請確保這裡列出的所有欄位名稱都存在於您的CSV檔案中，且沒有遺漏任何欄位。
# 否則程式可能會報錯。
desired_order = [
    "id",
    "created_at",
    "from_user_name",
    "from_user_id",
    "from_user_lang",
    "from_user_tweetcount",
    "from_user_followercount",
    "from_user_friendcount",
    "from_user_listed",
    "from_user_realname",
    "from_user_utcoffset",
    "from_user_timezone",
    "from_user_description",
    "from_user_url",
    "from_user_verified",
    "from_user_profile_image_url",
    "from_user_created_at",
    "from_user_withheld_scope",
    "from_user_favourites_count",
    "source",
    "location",
    "geo_lat",
    "geo_lng",
    "text",
    "retweet_id",
    "retweet_count",
    "favorite_count",
    "to_user_id",
    "to_user_name",
    "in_reply_to_status_id",
    "filter_level",
    "lang",
    "possibly_sensitive",
    "truncated",
    "withheld_copyright",
    "withheld_scope",
]

# 5. 檢查所有期望的欄位是否存在於 DataFrame 中
current_columns = df.columns.tolist()
missing_desired_columns = [col for col in desired_order if col not in current_columns]
extra_current_columns = [col for col in current_columns if col not in desired_order]

if missing_desired_columns:
    print(f"錯誤：在您的資料中找不到以下期望的欄位：{missing_desired_columns}")
    sys.exit(1)  # 如果有缺失欄位，則退出程式
if extra_current_columns:
    print(
        f"警告：您的資料中包含以下未在期望順序中列出的欄位，它們將被排除：{extra_current_columns}"
    )


# 6. 根據新的順序重新排序 DataFrame 的欄位
try:
    df_reordered = df[desired_order]
except KeyError as e:
    print(
        f"錯誤：在重新排序欄位時發生問題，可能是 'desired_order' 中的欄位名稱不正確：{e}"
    )
    sys.exit(1)

# 7. 將修改後的資料儲存回新的 CSV 檔案
try:
    df_reordered.to_csv(output_filepath, index=False)
    print(f"欄位位置已成功依照您的完整順序調整並儲存至 '{output_filepath}'。")
    print("\n調整後的資料預覽：")
    print(df_reordered.head())
except Exception as e:
    print(f"儲存檔案時發生錯誤：{e}")
    sys.exit(1)
