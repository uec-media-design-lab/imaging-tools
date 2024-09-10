import pandas as pd

img_path = "images/0909MTF_dis15-45/Result_csv"
csv_file_path = img_path + '/dis-15--deg-0--gl-0--mi-0.csv'  # CSVファイルのパス

# パラメータを設定
sensor_width_mm = 35  # センサーの横幅 [mm] α7Ⅲ → 35㎜
roi_px = 91770  # ROIの横幅 [ピクセル]

cam_sensor_dis = 30
focus_dis = 35

# ピクセルサイズを計算
pixel_size_mm = sensor_width_mm / roi_px  # 1ピクセルあたりの物理サイズ [mm]

# 1lp/mmをcycles/pixelに変換する関数
def lpmm_to_cycles_per_pixel(lp_mm, pixel_size_mm):
    return lp_mm * pixel_size_mm

#実際の撮像面の大きさ
pickup_size = (cam_sensor_dis / focus_dis) * sensor_width_mm

# 1lp/mmに相当するcycles/pixelを計算
lp_mm_threshold = 1  # 1 lp/mm
cycles_per_pixel_threshold = lpmm_to_cycles_per_pixel(lp_mm_threshold, pixel_size_mm)

# CSVファイルを読み込む（CSVファイルには 'cycles_per_pixel' と 'MTF_value' 列があると仮定）
data = pd.read_csv(csv_file_path)

# 1lp/mm以下に相当するcycles/pixelの範囲でフィルタリング
filtered_data = data[data['lp/mm'] <= lp_mm_threshold]
print(len(filtered_data))
#1lp/mmスケールのMTF
#lpmtf = data[data['cycles/px']]

# 平均MTF値を計算
average_mtf = filtered_data['lp/mm'].mean()

# 結果を表示
print(f"1lp/mm以下の平均MTF値: {average_mtf}")
