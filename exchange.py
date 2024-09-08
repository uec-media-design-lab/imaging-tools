import pandas as pd

img_path = "existif2/Result_csv"
csv_file_path = img_path + '/D=15.csv'  # CSVファイルのパス

# パラメータを設定
sensor_width_mm = 36  # センサーの横幅 [mm]
roi_px = 57820  # ROIの横幅 [ピクセル]

# ピクセルサイズを計算
pixel_size_mm = sensor_width_mm / roi_px  # 1ピクセルあたりの物理サイズ [mm]

# 1lp/mmをcycles/pixelに変換する関数
def lpmm_to_cycles_per_pixel(lp_mm, pixel_size_mm):
    return lp_mm * pixel_size_mm

# 1lp/mmに相当するcycles/pixelを計算
lp_mm_threshold = 1  # 1 lp/mm
cycles_per_pixel_threshold = lpmm_to_cycles_per_pixel(lp_mm_threshold, pixel_size_mm)

# CSVファイルを読み込む（CSVファイルには 'cycles_per_pixel' と 'MTF_value' 列があると仮定）
data = pd.read_csv(csv_file_path)

# 1lp/mm以下に相当するcycles/pixelの範囲でフィルタリング
filtered_data = data[data['cycles/px'] <= cycles_per_pixel_threshold]

# 平均MTF値を計算
average_mtf = filtered_data['mtf'].mean()

# 結果を表示
print(f"1lp/mm以下の平均MTF値: {average_mtf}")
