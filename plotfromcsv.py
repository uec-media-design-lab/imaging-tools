import matplotlib.pyplot as plt
import csv

# CSVファイルを読み込む (仮定: 各列が別のデータセットを表している)
csv_file = 'plot/plotdismtf2.csv'  # 実際のCSVファイル名を指定

# 各プロットに対応するリストを準備
floating_distance = []
Real_MMAP = []
Mitsuba_MMAP = []
Blender_MMAP = []
Ours_MMAP = []

# CSVファイルを読み込んでリストにデータを格納
with open(csv_file, 'r', newline='') as file:
    reader = csv.reader(file)
    next(reader)  # ヘッダーをスキップ
    for row in reader:
        floating_distance.append(float(row[0]))
        Real_MMAP.append(float(row[1]))
        Mitsuba_MMAP.append(float(row[2]))
        Blender_MMAP.append(float(row[3]))

# グラフを描画
plt.figure()

# 各データセットをプロット (マーカーや色を指定)
plt.plot(floating_distance, Real_MMAP, 'ko-', label='Real_MMAP')
#plt.plot(floating_distance, RF_AN, 'kD:', label='RF-AN')
plt.plot(floating_distance, Mitsuba_MMAP, 'bs-', label='Mitsuba_MMAP')
plt.plot(floating_distance, Ours_MMAP, 'bd:', label='Ours_MMAP')
plt.plot(floating_distance, Blender_MMAP, 'y^-', label='Blender_MMAP')
#plt.plot(floating_distance, Guo_RF_AN, 'y^:', label='Guo et al. (RF-AN)')

# 軸ラベルを設定
plt.xlabel('Floating distance [mm]')
plt.ylabel('Average of MTF [au]')

# 凡例を表示
plt.legend()

# グリッドを表示
plt.grid(True)

# グラフを表示
plt.show()