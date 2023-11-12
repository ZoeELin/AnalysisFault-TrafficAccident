import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

# 設定隨機種子
random.seed(42)
np.random.seed(42)

# 設定數據集大小
n_samples = 10000


# 事故時間
morning_peak_mean = 8  # 早高峰平均时间
morning_peak_std = 1.5  # 早高峰标准差
evening_peak_mean = 18  # 晚高峰平均时间
evening_peak_std = 1.5  # 晚高峰标准差

# 生成上下班高峰時間的事故時間數據
morning_peak_hours = np.random.normal(
    morning_peak_mean, morning_peak_std, int(n_samples / 2)
)
evening_peak_hours = np.random.normal(
    evening_peak_mean, evening_peak_std, int(n_samples / 2)
)

accident_hours = np.concatenate([morning_peak_hours, evening_peak_hours])  # 合并上下班高峰時間數據
accident_hours = np.clip(accident_hours, 0, 24)  # 限制時間在 0-24 小時之間
accident_hours = np.round(accident_hours).astype(int)  # 正數化
np.random.shuffle(accident_hours)  # 打亂數組順序


# 降雨量
rain_distribution = {
    "1-4": np.random.randint(0, 100, int(n_samples * 0.3 / 4)),  # 1-4月，0到99毫米
    "5-6": np.random.randint(0, 200, int(n_samples * 0.7 / 5)),  # 5-6月，0到199毫米
    "7-9": np.random.randint(100, 300, int(n_samples * 0.7 / 5)),  # 7-9月，100到299毫米
    "10-12": np.random.randint(0, 100, int(n_samples * 0.3 / 3)),  # 10-12月，0到99毫米
}

rainfall = np.concatenate(list(rain_distribution.values()))
if len(rainfall) < n_samples:
    # 少於數據量，用 0-99 填充
    additional_rainfall = np.random.randint(0, 100, n_samples - len(rainfall))
    rainfall = np.concatenate([rainfall, additional_rainfall])

np.random.shuffle(rainfall)  # 打亂數組順序


# 風速
wind_distribution = {
    "1-4": np.random.randint(0, 50, int(n_samples * 0.3 / 4)),  # 1-4 月，0到49公里/小时
    "5-6": np.random.randint(20, 100, int(n_samples * 0.7 / 5)),  # 5-6 月，20到99公里/小时
    "7-9": np.random.randint(50, 150, int(n_samples * 0.7 / 5)),  # 7-9 月，50到149公里/小时
    "10-12": np.random.randint(0, 50, int(n_samples * 0.3 / 3)),  # 10-12 月，0到49公里/小时
}

wind_speed = np.concatenate(list(wind_distribution.values()))
if len(wind_speed) < n_samples:
    # 如果生成的風速數據小於 1000，用 0-49 填充
    additional_wind = np.random.randint(0, 50, n_samples - len(wind_speed))
    wind_speed = np.concatenate([wind_speed, additional_wind])

np.random.shuffle(wind_speed)


# 駕駛速度
speed_mean = 45  # 平均時速 45km/hr
speed_std = 5  # 標準差

driving_speeds = np.random.normal(speed_mean, speed_std, n_samples)
driving_speeds = np.clip(driving_speeds, 0, 120)


# 酒精濃度數據
alcohol_levels = np.random.uniform(0.0, 0.12, n_samples)

# 違反交通信號
traffic_violations = np.random.choice(
    [0, 1], size=n_samples, p=[0.85, 0.15]
)  # 0的概率為85%，1的概率為15%

# 路寬
lane_width_mean = 3.5  # 平均車道寬度 3.5m
lane_width_std = 0.5  # 標準差

lane_widths = np.random.normal(lane_width_mean, lane_width_std, n_samples)
lane_widths = np.clip(lane_widths, 1.5, None)  # 限制路寬在 1.5m 以上

# 路面狀況
road_condition_mean = 3  # 平均路面状况
road_condition_std = 0.5  # 標準差

road_conditions = np.random.normal(road_condition_mean, road_condition_std, n_samples)
road_conditions = np.clip(road_conditions, 0, 5)  # 0-5

# 精神狀況
mental_state_prob = [0.65 / 7] * 7 + [
    0.35 / 4
] * 4  # 概率必須加起来等於1分配0-6的概率是0.65/7，大于6的是0.35/4
mental_states = np.random.choice(np.arange(0, 11), size=n_samples, p=mental_state_prob)

# 騎車技巧熟練
riding_skills = np.random.randint(0, 11, n_samples)

# 生成交通事故類別
accident_types = np.random.choice(
    [1, 2, 3], size=n_samples, p=[0.05, 0.25, 0.70]
)  # 概率分别調整為5%, 25%, 70%

data = {
    "accident_hours": accident_hours[:n_samples],
    "rainfall": rainfall[:n_samples],
    "wind_speed": wind_speed[:n_samples],
    "driving_speeds": driving_speeds,
    "alcohol_levels": alcohol_levels,
    "traffic_violations": traffic_violations,
    "lane_widths": lane_widths,
    "road_conditions": road_conditions,
    "mental_states": mental_states,
    "riding_skills": riding_skills,
    "accident_types": accident_types,
}


# 轉換為 DataFrame
df = pd.DataFrame(data)


# 根據規則設置過失標籤
df["falses"] = 0
df.loc[df["accident_types"] == 1, "falses"] = 1
df.loc[
    (df["traffic_violations"] == 1 & (df["alcohol_levels"] > 0.1))
    | (df["driving_speeds"] > df["lane_widths"] * 40),
    "falses",
] = 1
df.loc[
    (df["accident_hours"] >= 2) & (df["accident_hours"] <= 5) & (df["rainfall"] > 50),
    "falses",
] = 1
df.loc[(df["alcohol_levels"] > 0.1) & (df["lane_widths"] < 2), "falses"] = 1
df.loc[(df["driving_speeds"] > 70) & (df["lane_widths"] > 4), "falses"] = 1

# 將 "falses" 欄位添加到 data 之中一起繪製
data["falses"] = df["falses"]

# 創建畫布，配置為 4 行 3 列的子圖，整個畫布大小設置為 10x8 英寸
fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(10, 8))

# 遍歷 data，建立的直方圖
for ax, (feature, values) in zip(axes.flatten(), data.items()):
    ax.hist(values, bins=25, color="blue")  # bins=25 表示將數據分成 25 個區間
    ax.set_title(feature)

plt.tight_layout()
plt.show()


# 檢視過失為0的資料集
# no_fault_data = df.loc[df["falses"] == 0]

# 保存為 CSV 文件
# 儲存於當前路徑, 命名為 policy_data.csv
csv_file_path = "./inputs/policy_data.csv"
df.to_csv(csv_file_path, index=False)
