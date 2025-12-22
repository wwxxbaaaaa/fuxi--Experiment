# -*- coding: UTF-8 -*-
import pandas as pd
import os
import gzip
import json
import requests
import numpy as np
from tqdm import tqdm

# 1. 配置参数
DATASET_NAME = 'Amazon_Beauty'
BASE_PATH = os.path.join('data', DATASET_NAME)
# 使用斯坦福 SNAP 提供的 Amazon Beauty 5-core 数据
RAW_DATA_URL = "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Beauty_5.json.gz"
NEG_ITEM_NUM = 99

def download_data():
    if not os.path.exists(BASE_PATH):
        os.makedirs(BASE_PATH)
    file_path = os.path.join(BASE_PATH, 'reviews_Beauty_5.json.gz')
    
    if not os.path.exists(file_path):
        print(f"🚀 正在下载 Amazon Beauty 数据集 (约 10MB)...")
        try:
            r = requests.get(RAW_DATA_URL, stream=True)
            with open(file_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024): 
                    if chunk: f.write(chunk)
            print("✅ 下载完成！")
        except Exception as e:
            print(f"❌ 下载失败: {e}")
            print("请尝试手动下载：http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Beauty_5.json.gz")
            print("并将其放入 data/Amazon_Beauty/ 文件夹中。")
            exit()
    return file_path

def parse_gz(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield json.loads(l)

# --- 执行流程 ---

# 1. 下载与读取
zip_path = download_data()

print("📊 正在解析 JSON 数据...")
interactions = []
# 读取 Amazon 特有的 JSON 格式
for entry in tqdm(parse_gz(zip_path)):
    # 格式: reviewerID(用户), asin(商品), overall(评分), unixReviewTime(时间戳)
    interactions.append([entry['reviewerID'], entry['asin'], int(entry['unixReviewTime'])])

df = pd.DataFrame(interactions, columns=['user_id', 'item_id', 'time'])
print(f"原始交互数: {len(df)}")

# 2. 5-core 过滤 (确保数据质量)
print("正在执行 5-core 过滤...")
while True:
    u_counts = df.groupby('user_id').size()
    i_counts = df.groupby('item_id').size()
    df = df[df.user_id.isin(u_counts[u_counts >= 5].index) & df.item_id.isin(i_counts[i_counts >= 5].index)]
    if (u_counts < 5).sum() == 0 and (i_counts < 5).sum() == 0: break
print(f"过滤后交互数: {len(df)}")

# 3. ID 重映射
print("🆔 正在映射 ID...")
user2new = {old: i + 1 for i, old in enumerate(sorted(df.user_id.unique()))}
item2new = {old: i + 1 for i, old in enumerate(sorted(df.item_id.unique()))}
df['user_id'] = df['user_id'].map(user2new)
df['item_id'] = df['item_id'].map(item2new)

# 4. 构建全局点击字典 (解决 KeyError 的核心)
clicked_item_set = df.groupby('user_id')['item_id'].apply(set).to_dict()
all_items_array = np.array(list(item2new.values()))

# 5. 切分数据集 (Leave-One-Out)
# 策略：每个用户的最后一次购买做 Test，倒数第二次做 Dev，其余做 Train
print("✂️ 正在切分数据集 (Leave-One-Out)...")
df = df.sort_values(['user_id', 'time'])
train_data, dev_data, test_data = [], [], []

for uid, group in tqdm(df.groupby('user_id')):
    items = group['item_id'].tolist()
    times = group['time'].tolist()
    
    if len(items) < 3: continue 
    
    # Train: 除去最后两条
    for i in range(len(items) - 2):
        train_data.append([uid, items[i], times[i]])
    # Dev: 倒数第二条
    dev_data.append([uid, items[-2], times[-2]])
    # Test: 最后一条
    test_data.append([uid, items[-1], times[-1]])

train_df = pd.DataFrame(train_data, columns=['user_id', 'item_id', 'time'])
dev_df = pd.DataFrame(dev_data, columns=['user_id', 'item_id', 'time'])
test_df = pd.DataFrame(test_data, columns=['user_id', 'item_id', 'time'])

# 6. 负采样
print("🎲 正在生成负样本 (每条 99 个)...")
def generate_negative(data_df, seed):
    np.random.seed(seed)
    neg_items = []
    uids = data_df['user_id'].values
    for uid in tqdm(uids):
        user_clicked = clicked_item_set[uid]
        negs = []
        while len(negs) < NEG_ITEM_NUM:
            n = np.random.choice(all_items_array)
            if n not in user_clicked and n not in negs:
                negs.append(int(n))
        neg_items.append(str(negs)) # 保存为列表字符串
    return neg_items

dev_df['neg_items'] = generate_negative(dev_df, seed=1)
test_df['neg_items'] = generate_negative(test_df, seed=2)

# 7. 保存
print(f"💾 正在保存文件到 {BASE_PATH}...")
cols = ['user_id', 'item_id', 'time']
train_df[cols].to_csv(os.path.join(BASE_PATH, 'train.csv'), sep='\t', index=False)
dev_df[cols + ['neg_items']].to_csv(os.path.join(BASE_PATH, 'dev.csv'), sep='\t', index=False)
test_df[cols + ['neg_items']].to_csv(os.path.join(BASE_PATH, 'test.csv'), sep='\t', index=False)

print("✨ Amazon Beauty 数据集处理完成！")