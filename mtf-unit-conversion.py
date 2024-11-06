import sys
import csv

''' 以下を編集する '''
focal_length_mm = 50                # カメラの焦点距離 (mm)
horizontal_resolution_px = 9568     # カメラの横方向の解像度 (px)
horizontal_sensor_size_mm = 36      # カメラのイメージセンサの横の長さ (mm)
camera_edge_distance_mm = 570       # カメラから撮影対象（エッジ画像）までの距離 (mm)



''' 以下は基本的に編集不要（間違いがあったら更新してください） '''

input_file = sys.argv[1]                   # slanted-edge-mtf.py が出力したCSVファイルを指定する
output_file = input_file + '_lp_mm.csv'    # 書き込み用のCSVファイル名

def convert_unit(cycles_px):

    imaging_size_mm = (camera_edge_distance_mm / focal_length_mm) * horizontal_sensor_size_mm
    lp_mm = cycles_px * horizontal_resolution_px / imaging_size_mm

    return lp_mm


# Copilot生成プログラムを編集
with open(input_file, mode='r', newline='', encoding='utf-8') as infile, \
     open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    
    # ヘッダーを読み込み、書き込み
    headers = next(reader)
    headers.append('lp/mm')    # 3列目のヘッダーを追加
    writer.writerow(headers)

    # 各行を処理
    for row in reader:
        cycles_px = float(row[0])          # 2列目の値を取得
        lp_mm = convert_unit(cycles_px)    # lp_mm を計算
        row.append(lp_mm)                  # 3列目に追加
        writer.writerow(row)               # ファイルに書き込み

print(f"{output_file} generated.")
