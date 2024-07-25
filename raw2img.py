import rawpy
import sys
import os
import cv2
import tqdm    # 進捗バー表示

# ディレクトリ作成
def mkdir(directoryPath):
   if not os.path.exists(directoryPath):
      os.makedirs(directoryPath)
   return directoryPath

# 指定フォルダ内の特定の拡張子のリストを作成して返す
def mkfilelist(folderpath, ext):
   files = os.listdir(folderpath)
   res = []    # ファイルの全体パス (例：./folder/file.ext  => ./folder.file.ext)
   res2 = []   # ファイル名         (例：./folder/file.ext  => file)
   for f in files:
      fbase, fext = os.path.splitext(f)
      if fext=="."+ext:
         res.append(folderpath + "/" + fbase + fext)
         res2.append(fbase)
   return res, res2

# RAWデータ(*.ARW)を任意の画像データに変換(cv2の対応しているフォーマット)
# 16bit/ch, ガンマ補正無しで変換

# raw2img.py [RAWデータを格納してるフォルダ] [出力先フォルダ] [形式]
# ex: raw2img.py rawFolder tifFolder tif
def main():
   if len(sys.argv)<4 or os.path.isfile(sys.argv[1]):
         # 引数不足か入力がフォルダではない
         print("Usage: raw2img.py rawFolder outputFolder extension")
         sys.exit(-1)
   
   # print(sys.argv)
   inputDirectory = sys.argv[1]
   outputDirectory = sys.argv[2]
   ext = sys.argv[3]
   
   mkdir(outputDirectory)

   files, fileNames = mkfilelist(inputDirectory, 'ARW')
   #forを使って読み込んだファイルを１個１個処理
   # for i in range(len(files)):    # 進捗バー表示しなくていいなら
   for i in tqdm.tqdm(range(len(files)), leave=False):
      outputFilePath = outputDirectory + '/' + fileNames[i] + '.' + ext
      # print(outputFilePath)
      
      rawData = rawpy.imread(files[i])
      # postprocessでnumpyのarrayにデータを詰め込める
      # オプション参考：https://letmaik.github.io/rawpy/api/rawpy.Params.html#rawpy.Params
      # use_camera_wb:  カメラのホワイトバランス設定の適用
      # no_auto_bright: 輝度自動補正
      # gamma:          ガンマ補正
      # output_bps:     出力ファイルのbit数
      rgb = rawData.postprocess(use_camera_wb=True,no_auto_bright=False, gamma=(1,1), output_bps=16)

      # OpenCV2のimwriteで保存。この関数はBGR(RGBではない)の順で書き込むらしい
      # のでcvtColor関数で変換挟む
      cv2.imwrite(outputFilePath, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
      rawData.close()
      
   
if __name__ == "__main__":
   main()