# get neg edge

import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from PIL import Image
import os
from tqdm import tqdm

# 路径配置
RADIO_MAP_BASE_PATH = "/home/DataDisk/qmzhang/RadioMapSeer"
TX_PATH = f"{RADIO_MAP_BASE_PATH}/png/antennas"      
BUILDING_PATH = f"{RADIO_MAP_BASE_PATH}/png/buildings_complete"

# 定义要处理的数据集
DATASETS = [
    {
        "name": "IRT4",
        "gain_path": f"{RADIO_MAP_BASE_PATH}/gain/IRT4",
        "output_path": f"{RADIO_MAP_BASE_PATH}/gain/IRT4_k2_neg_norm"
    },
    {
        "name": "DPM", 
        "gain_path": f"{RADIO_MAP_BASE_PATH}/gain/DPM",
        "output_path": f"{RADIO_MAP_BASE_PATH}/gain/DPM_k2_neg_norm"
    },
    {
        "name": "carsDPM",
        "gain_path": f"{RADIO_MAP_BASE_PATH}/gain/carsDPM", 
        "output_path": f"{RADIO_MAP_BASE_PATH}/gain/carsDPM_k2_neg_norm"
    }
]

# 为所有输出路径创建目录
for dataset in DATASETS:
    os.makedirs(dataset["output_path"], exist_ok=True)
# 参数配置
min_map, max_map = 0, 701
min_Tx, max_Tx = 0, 80
bounding_threshold = 61
PathLoss_trunc = -147 # db
PathLoss_max = -47 # db
source_power = 23 # dbm
h = 1.00
one_over_h2 = 1 / h**2
p = 0.01


# 计算总任务数 (3个数据集 × 每数据集的任务数)
total_tasks = len(DATASETS) * (max_map - min_map) * (max_Tx - min_Tx)

# 使用tqdm显示进度条
with tqdm(total=total_tasks, desc="处理所有数据集", unit="张") as pbar:
    # 遍历每个数据集
    for dataset in DATASETS:
        print(f"\n开始处理数据集: {dataset['name']}")
        
        # 为当前数据集创建进度条
        dataset_tasks = (max_map - min_map) * (max_Tx - min_Tx)
        with tqdm(total=dataset_tasks, desc=f"处理{dataset['name']}", unit="张", leave=False) as dataset_pbar:
            for map_index in range(min_map, max_map):
                for Tx_index in range(min_Tx, max_Tx):
                    # get the path of the image
                    # 随机选择p概率是否继续处理还是continue
                    # print("map_index:",map_index,"Tx_index:",Tx_index)
                    image_path = f"{dataset['gain_path']}/{map_index}_{Tx_index}.png"
                    # get the path of the Tx image  
                    tx_path = f"{TX_PATH}/{map_index}_{Tx_index}.png"
                    # get the path of the building image
                    building_path = f"{BUILDING_PATH}/{map_index}.png"
                    # show gray image
                    image_gain = np.asarray(io.imread(image_path), dtype=np.float32) / 255
                    image_Tx = np.asarray(io.imread(tx_path), dtype=np.float32) / 255
                    building = np.asarray(io.imread(building_path), dtype=np.float32) / 255
                    
                    PathLoss_scale = image_gain
                    PathLoss_db = PathLoss_trunc + (PathLoss_max - PathLoss_trunc) * PathLoss_scale
                    # print("PathLoss_db:",PathLoss_db)
                    # 将光源强度单位从dbm换算为db
                    source_power_db = source_power - 30
                    # 再根据光源强度(db)与P_L(db)得到所有位置的实际光源强度(W), pathloss_db 为负数
                    power_db = np.ones_like(PathLoss_db) * source_power_db + PathLoss_db
                    power_w = 10 ** (power_db / 10)
                    # print("power_w:",power_w)
                    u = power_w
                    delta_u = (u[2:, 1:-1] + u[:-2, 1:-1] + u[ 1:-1, 2:] + u[1:-1, :-2]) - 4 * u[1:-1, 1:-1]
                    delta_u = delta_u * one_over_h2
                    # print("delta_u:",delta_u)
                    # image_Tx = image_Tx
                    tmp = (delta_u + image_Tx[1:-1, 1:-1] * u[1:-1, 1:-1]) / u[1:-1, 1:-1]
                    # print(np.where(image_Tx[1:-1, 1:-1] >= 1.0))
                    # tmp = np.abs(tmp)
                    k2_neg = np.where(tmp < 0, tmp, 0.)
                    # print("tmp:",tmp)
                    k2_neg_norm = (k2_neg - np.min(k2_neg)) / (np.max(k2_neg) - np.min(k2_neg))
                    # 归一化k到0到1
                    # k = k - np.min(k) / ( np.max(k) - np.min(k) )
                    # 将k转换为灰度图 并和image_gain原图并排绘制 分别给出标题   
                    k2_neg_norm = k2_neg_norm * 255
                    k2_neg_norm = k2_neg_norm.astype(np.uint8)
                    k2_neg_norm = np.pad(k2_neg_norm, ((1,1), (1,1)), mode='constant', constant_values=255)

                   
                    Image.fromarray(k2_neg_norm).save(f"{dataset['output_path']}/{map_index}_{Tx_index}.png")
                    
                    # 更新进度条
                    dataset_pbar.update(1)
                    pbar.update(1)
        
        print(f"完成处理数据集: {dataset['name']}")
        
        