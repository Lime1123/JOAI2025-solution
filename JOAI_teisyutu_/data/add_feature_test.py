import pandas as pd
import numpy as np
import cv2
import os
from pathlib import Path

# データの読み込み
df = pd.read_csv('data/test_temp.csv')

# 摂氏からケルビンへの変換関数
def celsius_to_kelvin(c):
    return c + 273.15

# 華氏からケルビンへの変換関数
def fahrenheit_to_kelvin(f):
    return (f - 32) * 5/9 + 273.15

# 温度の単位に応じて絶対温度を計算
df['absolute_temp_min'] = np.where(
    df['temp_unit'] == 'C',
    celsius_to_kelvin(df['temp_min']),
    fahrenheit_to_kelvin(df['temp_min'])
)

df['absolute_temp_max'] = np.where(
    df['temp_unit'] == 'C',
    celsius_to_kelvin(df['temp_max']),
    fahrenheit_to_kelvin(df['temp_max'])
)

# サーマルイメージの統計情報を取得する関数
def get_thermal_image_stats(image_path):
    """サーマル画像の統計情報を取得する"""
    try:
        # 画像パスを構築する
        full_path = os.path.join('data/thermal_images', image_path)
        # 画像が存在するか確認
        if not os.path.exists(full_path):
            return None, None, None, None, None, None, None
        
        # 画像を読み込む
        img = cv2.imread(full_path)
        if img is None:
            return None, None, None, None, None, None, None
        
        # グレースケールに変換（サーマル画像の温度値を近似）
        gray = img#cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 統計情報を計算
        img_min = np.min(gray)
        img_max = np.max(gray)
        img_avg = np.mean(gray)
        img_std = np.std(gray)
        img_q1 = np.percentile(gray, 25)
        img_q2 = np.percentile(gray, 50)  # メディアン
        img_q3 = np.percentile(gray, 75)
        
        return img_min, img_max, img_avg, img_std, img_q1, img_q2, img_q3
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None, None, None, None, None, None, None

# 画像パスから統計情報を取得して新しい列を追加
print("サーマルイメージの統計情報を抽出中...")
stats_data = []

for image_path in df['image_path_uuid']:
    stats = get_thermal_image_stats(image_path)
    stats_data.append(stats)

# 抽出した統計情報をデータフレームに追加
df['thermal_image_min'], df['thermal_image_max'], df['thermal_image_avg'], \
df['thermal_image_std'], df['thermal_image_q1'], df['thermal_image_median'], \
df['thermal_image_q3'] = zip(*stats_data)

# 結果の保存
df.to_csv('data/test_temp.csv', index=False)
print("処理完了: サーマルイメージの統計情報を追加しました")
