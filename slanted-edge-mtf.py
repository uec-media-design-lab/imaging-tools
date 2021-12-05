#!/usr/bin/python3 -B
import os                        # built-in module
import sys                       # built-in module
import time                      # built-in module
import json                      # built-in module
import pprint                    # built-in module
import cv2                       # pip install opencv-python
import imgio                     # pip install imgio
import numpy as np               # pip install numpy
import scipy.signal              # pip install scipy
import scipy.ndimage.morphology  # pip install scipy
import matplotlib.pyplot as pp   # pip install matplotlib
import matplotlib.widgets        # pip install matplotlib
import argv                      # local import
import csv

DEBUG = False

EDGE_WIDTH = 99
MIN_ROI_WIDTH = 100
MIN_ROI_HEIGHT = 50
MIN_ROI_SIZE = (MIN_ROI_HEIGHT, MIN_ROI_WIDTH)


DEFAULT_CONFIG = {
    "roi-center": [],  # [miny, maxy, minx, maxx]
    "roi-top-left": [],
    "roi-top-right": [],
    "roi-bottom-left": [],
    "roi-bottom-right": [],
    "edge-width": EDGE_WIDTH,
    "edge-min-angle": 78,
    "edge-max-angle": 88,
}

# MTF解析結果をひとつのクラスにまとめて管理
class MTFResults(object):

    def __init__(self, corner):
        self.corner = corner         # center|top-left|...
        self.esf = None              # Edge Spread Function         # ESF
        self.lsf = None              # Line Spread Function         # LSF
        self.lsfs = None             # smoothed LSF                 # wienerフィルタでスムージングしたLSF
        self.mtf = None              # FFT of raw LSF               # LSFを基に計算したMTF
        self.mtfs = None             # FFT of smoothed LSF          # LSFSを基に計算したMTF。スムージングした結果
        self.mtf50 = None            # MTF50 in cycles/pixel        # MTFが0.5を指す空間周波数
        self.mtf20 = None            # MTF20 in cycles/pixel        # MTFが0.2を指す空間周波数
        self.edge_straight = None    # straightened edge region     # 斜めの線を直線にそろえたときのエッジデータ
        self.edge_region = None      # edge region source pixels
        self.edge_mask = None        # edge region binary mask
        self.edge_coords = None      # edge y-coords & x-coords
        self.edge_coeffs = None      # edge coeffs: y = ax + b
        self.edge_angle = None       # edge angle in degrees
        self.edge_yspan = None       # edge length in pixels
        self.success = False         # True|False

    def report(self):
        if self.success:    # 成功時：エッジ角、エッジの高さ(？)、MTF50と20の結果を出力
            print("-" * 60)
            print("Results for {} region:".format(self.corner))
            print("  Edge angle: {:.1f} degrees".format(self.edge_angle))
            print("  Edge height: {} pixels".format(self.edge_yspan))
            print("  MTF50: {:.3f} cycles/pixel = {:.1f} pixels/cycle".format(self.mtf50, 1.0 / self.mtf50))
            print("  MTF20: {:.3f} cycles/pixel = {:.1f} pixels/cycle".format(self.mtf20, 1.0 / self.mtf20))
        else:
            print("-" * 60) # 失敗時：失敗したことを出力
            print("MTF calculation for {} region failed.".format(self.corner))

# MTFの本体的関数
def mtf(config, results, filename, outputDir):
    print("Configuration:")
    pprint.pprint(config, indent=2) # configの内容をインデント幅2で、改行して見やすく表示している
    min_angle = config["edge-min-angle"]    # config(辞書型変数)の   edge-min-angleを取得
    max_angle = config["edge-max-angle"]    #                       edge-max-angle
    edge_width = config["edge-width"]       #                       edge-width

    # read source image, convert to grayscale, normalize [black, white] = [0, 1]
    # 画像読み込み、グレスケ変換、正規化(黒--白を0--1の小数で)
    basename = os.path.basename(filename)       # ファイル名取得(拡張子あり)
    barename = os.path.splitext(basename)[0]    # ファイル名    (拡張子なし)
    image = imread(filename)                # 自作の画像読み込み関数
    imgh, imgw = image.shape                # 縦横サイズ

    # plot source image - selected regions will be overlaid later
    # ソース画像表示用のpp用意？ ROIが上書きで描画されるらしい。
    fig, axis = pp.subplots(num="image", nrows=1, ncols=1, squeeze=True, figsize=(17,9), dpi=110)
    fig.canvas.set_window_title("slanted-edge-mtf: Selected regions")   # ウィンドウ名セット
    axis.imshow(image)  # 画像表示
    pp.title(basename)  # タイトルセット

    for idx, res in enumerate(results): # enumerate: インデックス番号,要素を同時に代入できる関数。便利

        # extract region of interest
        # ROIの抽出
        prefix = "{} edge detection failed".format(res.corner)  # prepended to all error messages   エラー文用意？
        key = "roi-{}".format(res.corner)   # corner: 中央、左上、右上など。MTFResultsクラス
        roi = np.array(config[key])
        if roi is None or len(roi) == 0:    # roiが選択されてない場合
            print("{} region not specified, skipping...".format(res.corner))    # スキップ
            continue
        roi_valid = np.all((roi >= 0) & (roi < [imgh, imgh, imgw, imgw]))   # 有効な場合：ROIがあって画像サイズより小さい
        enforce(roi_valid, "{0}: Selected region {1} is exceeding image boundaries ({3} x {2})."
                .format(prefix, roi, *image.shape))     # 直前行がFalseの場合エラー
        region = image[roi[0]:roi[1], roi[2]:roi[3]]    # ROIで切り取り
        roih, roiw = region.shape                       # ROIのサイズ取得 => roih, roiw
        roi_valid = np.all(region.shape > MIN_ROI_SIZE) # 有効な場合：ROIサイズが冒頭で定義したサイズ以上
        err_plotter = lambda: plot_edge([region], suptitle=res.corner)  # 引数なしのラムダ式。plot_edgeを引数[region]で呼ぶ。(ラムダ式の意味ある？)
        axis.add_patch(pp.Rectangle((roi[2], roi[0]), roiw, roih, edgecolor="red", facecolor="none"))   # グラフにROI領域の描画処理を追加
        enforce(roi_valid, "{0}: Selected region must be at least {2} x {1} pixels; was {4} x {3}."
                .format(prefix, *MIN_ROI_SIZE, *region.shape), err_plotter) # ROIサイズが有効じゃない場合エラー

        # detect edge pixels
        # エッジ抽出
        #   大津のアルゴリズムで二値化                  =>  otsu_map
        #   morpho関数で小さなノイズ除去                =>  otsu_filt
        #   cannyフィルタで一意にエッジ出す             =>  otsu_edges
        #   np.nonzeroで非ゼロ要素(エッジの座標)抽出    =>  edge_coords     co-ordinate: 座標
        #   エッジとなりえるピクセルの数(？)カウント     =>  edge_y_span
        otsu_map = otsu(region)                               # generate binary mask: 0=black, 1=white
        otsu_filt = morpho(otsu_map)                          # filter out small non-contiguous regions
        otsu_edges = canny(otsu_filt)                         # detect edges; there should be only one
        edge_coords = np.nonzero(otsu_edges)                  # get (x, y) coordinates of edge pixels
        edge_y_span = len(np.unique(edge_coords[0]))          # get number of scanlines intersected
        err_plotter = lambda: plot_edge([region, otsu_map, otsu_filt, otsu_edges], suptitle=res.corner)     # 引数なしのラムダ式。plot_edgeを引数[region, otsu_map, ...]で呼ぶ。(ラムダ式の意味ある？)
        enforce(edge_y_span > MIN_ROI_HEIGHT, "{}: Edge must have at least {} scanlines; had {}."
                .format(prefix, MIN_ROI_HEIGHT, edge_y_span), err_plotter)  # edge_y_spanがROI最小値以下の時エラー

        # fit a straight line through the detected edge
        # エッジポイントから直線フィッティング
        edge_coeffs = np.polyfit(*reversed(edge_coords), deg=1)     # エッジの点群から直線フィッティング。  deg=1: 1次近似  reversed:リスト反転
        plot_edge([region, otsu_map, otsu_filt, otsu_edges], edge_coeffs, suptitle=res.corner)  # plot_edgeを呼ぶ(デバッグ時に動作)。各種フィルタリングの結果や得られた直線を描画する？
        edge_angle = np.abs(np.rad2deg(np.arctan(edge_coeffs[0])))  # aran→角度計算 rad2deg→単位degree  abs→絶対値で一意に角度決める
        enforce(min_angle < edge_angle < max_angle, "{}: Edge angle must be [{}, {}] degrees; was {:.1f}."
                .format(prefix, min_angle, max_angle, edge_angle))  # 角度が指定の範囲を逸脱した場合エラー
        prompt("Review the {} edge plots, then press Enter to continue.".format(res.corner.lower()))    # デバッグ時：「エッジ描画した」「Enterで次へ」

        # extract EDGE_WIDTH pixels from each scanline, centered on the detected edge
        # 検出したエッジの閾値を中心に、EDGE_WIDTHピクセルだけスキャンする
        # poly1d: 1字多項式(数式リスト, 変数記号)   poly1d([1,2,3], variable="X") => 1X^2 + 2X + 3
        # poly1d.c: 多項式の係数    c[0]で定数項
        px = np.poly1d(edge_coeffs, variable="x")  # y = ax + b  <==> x = (y - b) / a
        py = np.poly1d([1.0 / px.c[0], -px.c[1] / px.c[0]], variable="y")   # これら2行で、y=の形からx=の形を導いてる
        xp = np.polyval(py, np.arange(0, roih))  # ideal edge position on each scanline     # np.arange:等差数列の生成。0--ROI高さまでのリスト      polyval(数式, 評価するための式や値) 要するに代入操作    roih個の値が出てくる
        xpi = xp.round().astype(np.int32)   # (それぞれのエッジポイントに対して)小数切り捨て→int変換。エッジ中心のx座標[px]が得られる
        xfirst = xpi - edge_width // 2      # 中心からedge_width/2引いた位置が始点(移動幅の小数は切り捨て)      //2: 切り捨て除算
        xlast = xpi + edge_width // 2       # 中心からedge_width/2足した位置が終点(移動幅の少数派切り捨て)
        valid_rows = (xfirst >= 0) & (xlast < roiw)     # 有効な範囲に収まっているかチェック。始点・終点が画像内にあるかどうか
        xfirst = xfirst[valid_rows]
        xlast = xlast[valid_rows]
        enforce(np.sum(valid_rows) >= MIN_ROI_HEIGHT, "{}: Edge must have at least {} valid scanlines; had {}."
                .format(prefix, MIN_ROI_HEIGHT, np.sum(valid_rows)))    # 有効なスキャンライン数が一定以下の場合エラー
        xmin = np.min(xfirst)   # xmin:始点x座標の最小値
        xmax = np.max(xlast)    # xmax:終点x座標の最大値
        xfirst -= xmin          # 原点ずらし
        xlast -= xmin           # 原点ずらし
        crop = region[valid_rows, xmin:xmax+1]      # 解析に使う領域をクリッピング  縦：有効な行      横：始点最小値--終点最大値
        roih, roiw = crop.shape     # ROIを有効な範囲に限定して更新
        edge_straight = np.zeros((roih, edge_width), dtype=np.float32)  # float32型、縦ROI高さ、横エッジ幅のゼロ行列生成
        edge_straight[:] = [crop[y, xfirst[y]:xlast[y]+1] for y in range(roih)]     # 代入操作。スキャンラインごとにそれぞれの値格納。
        # 12345__
        # _67890_
        # __abcde
        
        # => ┌ 12345 ┐
        #    │ 67890 │
        #    └ abcde ┘

        # store results
        # 結果の格納。MTFResultsクラスのresultsのリストの要素、resのメンバー変数にそれぞれの結果を代入する
        res.edge_straight = edge_straight   # 斜めったエッジを縦に戻した時の各信号地
        res.edge_region = region            # ROIで切り取った画像片
        res.edge_mask = otsu_filt           # 大津二値化+細かなノイズ除去の結果
        res.edge_coeffs = edge_coeffs       # 直線フィッティングの結果
        res.edge_angle = edge_angle         # エッジの角度
        res.edge_yspan = roih               # ROI内の有効なエッジ高さ(スキャンライン数)

    pp.close("edges")

    # それぞれのresultに対してESF, LSF, MTF計算する
    for idx, res in enumerate(results):     # enumerate: インデックス番号,要素を同時に代入できる関数。
        if res.edge_straight is not None:   # edge_straight(解析対象)がある時に解析
            # compute Edge Spread Function (ESF), Line Spread Function (LSF), and filtered LSF
            edge = res.edge_straight
            res.esf = esf = np.mean(edge, axis=0)               # 行列のy方向に平均とってESFを出す
            res.esf = esf = scipy.signal.wiener(esf, 5)[3:-3]   # wienerフィルタでノイズ除去
            res.lsf = lsf = np.gradient(esf)[1:]                # 勾配計算(微分)してLSFを出す
            res.lsfs = lsfs = scipy.signal.wiener(lsf, 7)[4:-4] # wienerフィルタ重ね掛けでノイズ除去 => LSFS
            plot_lsf([edge], [esf, lsf, lsfs], ["Edge Profile", "LSF", "Filtered LSF"], res.corner)     # デバッグ時：ESF, LSF, LSFSをプロット
            prompt("Review the {} ESF & LSF curves, then press Enter to continue.".format(res.corner))          # デバッグ時：プロンプト上にテキスト表示+Enterで先に進む
            # compute filtered & unfiltered MTF
            res.mtf = mtf = fft(lsf)        # LSF を元にMTF 計算
            res.mtfs = mtfs = fft(lsfs)     # LSFSを元にMTFS計算
            # compute MTF50 & MTF20 from filtered MTF
            x_mtf = np.linspace(0, 1, len(mtf))     # 0--1の等差数列生成(mtfのデータ数で等分)
            # np.interp(x, xp, fp): 線形補完する関数
            #   x: 評価するX座標(内挿、xpの中にある必要があり)
            #   xp:X軸(単調増加)
            #   fp:Y軸の値
            # np.interp(2.5, [1,2,3], [3,2,0]) => 1.0
            #   2.5は2と3の中間。fpの対応する点は2と0なので、(2+0)/2 = 1.0
            res.mtf50 = mtf50 = np.interp(0.5, mtfs[::-1], x_mtf[::-1])     # 線形補完して、MTFの値が0.5になる空間周波数を求めている
            res.mtf20 = mtf20 = np.interp(0.2, mtfs[::-1], x_mtf[::-1])     # 線形補完して、MTFの値が0.2になる空間周波数を求めている
            res.success = True

    pp.close("lsf")
    
    #########################
    # mkdir
    mkdir(outputDir + "/Result_fig")
    mkdir(outputDir + "/Result_csv")
    #########################

    # それぞれの結果に対してMTFグラフのプロットと結果出力をする
    for idx, res in enumerate(results):     # enumerate: インデックス番号,要素を同時に代入できる関数
        if res.success:                     # 解析成功して無事MTFが計算できたときに実行
            label = "{}: MTF50 = {:.3f} cycles/pixel = {:.1f} pixels/cycle".format(res.corner, res.mtf50, 1.0 / res.mtf50)  # MTF値が0.5になる点の空間周波数をラベル
            ### save_mtfData
            # 改良部：MTFと空間周波数の関係をCSVファイルに出力
            csvName = outputDir + "/Result_csv/{}.csv".format(barename)
            data = [np.linspace(0,1,len(res.mtfs)), res.mtfs]
            data = list(zip(*data))# 転置
            file = open(csvName, 'w')
            writer = csv.writer(file)
            writer.writerow(["cycles/px", "mtf"])
            writer.writerows(data)
            file.close()
            ###
            
            plot_mtf(res.mtfs, res.mtf50, res.mtf20, label=label, color=pp.cm.cool(idx / 4))    # MTFグラフの描画(スムージングしたもの)
            if DEBUG:  # plot the unfiltered MTF only in debug mode
                plot_mtf(res.mtf, res.mtf50, res.mtf20, color=pp.cm.cool(idx / 4), linestyle=":", linewidth=0.5)    # デバッグ時はスムージング処理を施していないMTF曲線も描画する

    roi_filename = outputDir + "/Result_fig/" + "{}-ROI.png".format(barename)   # ROI描画グラフの保存先
    lsf_filename = outputDir + "/Result_fig/" + "{}-LSF.png".format(barename)   # LSF描画グラフの保存先
    mtf_filename = outputDir + "/Result_fig/" + "{}-MTF.png".format(barename)   # MTF描画グラフの保存先
    pp.title("MTF - {}".format(basename))   # タイトル設定
    pp.show(block=False)                    # 表示
    pp.figure("mtf")                        # plot_mtf関数で用意したMTF曲線のグラフを用意(デバッグモードではスムージングの代わりに生MTF曲線が代わりに保存される)
    pp.savefig(mtf_filename)                # MTFグラフの保存
    pp.figure("image")                      # mtf関数のはじめの方で用意したグラフを指定
    pp.savefig(roi_filename)                # ROI描画したグラフの保存
    # (LSF曲線はデフォで保存してない。デバッグ時は"curves"の名でグラフ描画しているので、欲しくなったらそれを使う)
    success = np.all([res.success for res in results])
    return success

# ファイル読み込み
def imread(filename, verbose=True):
    # imgio参考：https://github.com/toaarnio/imgio/blob/master/imgio/imgio.py
    # verbose：Trueにすると色々ログ出力するっぽい
    image, maxval = imgio.imread(filename, verbose=verbose)
    image = np.dot(image, [0.2125, 0.7154, 0.0721])  # RGB => Luminance
    
    # LuminanceとY値は微妙に違うらしい。
    # RGB => Luminance:     0.2125*R + 0.7154*G + 0.0721*B  
    # RGB => Y:             0.299*R + 0.587*G + 0.114*B     ITU-RBT.709(Rec.709)より
    
    # ここで輝度正規化している。
    image = image / maxval
    image = normalize(image)
    return image

# 画像の正規化
def normalize(image):
    # percentile関数で最小値と最大値を取得
    black = np.percentile(image, 0.1)       # ソートしたときの下位0.1%にあたる値をblackとする
    white = np.percentile(image, 99.9)      # ソートしたときの下位99.9%(上位0.1%)の値をwhite
    image = (image - black) / (white - black)   # コントラスト比の計算？
    image = np.clip(image, 0, 1)    # np.clip: 第二引数--第三引数の範囲に収まるようスケール変換
    return image

# 大津の二値化アルゴリズムを利用
# (8bitで処理してる？)
def otsu(image):
    # Otsu's binary thresholding
    # ガウスぼかし：ノイズ低減
    image = cv2.GaussianBlur(image, (5, 5), 0)  # simple noise removal
    image = (image * 255).astype(np.uint8)      # [0, 1] => [0, 255]
    otsu_thr, otsu_map = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)  # 大津の二値化
    return otsu_map

# 二値化画像に対しmorphologicalフィルタリング
# 要するにノイズ除去
def morpho(mask):
    # morphological filtering of binary mask: 3 x (erosion + dilation)
    structure = np.ones((3,3))  # 8-connected structure
    mask = scipy.ndimage.morphology.binary_opening(mask, structure, iterations=3)
    return mask

# cannyフィルタ。エッジ検出に使う？
def canny(image):
    # Canny edge detection
    image = (image * 255).astype(np.uint8)  # [0, 1] => [0, 255]
    edge_map = cv2.Canny(image, image.min(), image.max(), apertureSize=3, L2gradient=True)
    return edge_map

# フーリエ変換
def fft(lsf):
    # FFT of line spread function
    # 実際はフーリエ変換後に絶対値とって正規化している：つまりLSFからMTF計算するプロセスをここで完結させてる
    fft = np.fft.fft(lsf, 1024)  # even 256 would be enough     # np.fft.fft(データ, 個数(オプション))  256で十分らしいが…
    fft = fft[:len(fft) // 2]    # drop duplicate half          # 半分にカット
    fft = np.abs(fft)            # |a + bi| = sqrt(a² + b²)     # 絶対値を取得
    fft = fft / fft.max()        # normalize to [0, 1]          # 正規化
    return fft

# MTFグラフの描画
def plot_mtf(mtf, mtf50, mtf20, **kwargs):  # MTFデータ, MTF0.5の空間周波数, MTF0.2の空間周波数, その他オプション
    fig = pp.figure(num="mtf", figsize=(17,9), dpi=110)         # 名前：mtf, 縦横比17:9, dpi110
    fig.canvas.set_window_title("slanted-edge-mtf: MTF curves") # ウィンドウ名セット
    pp.grid(linestyle=":")                                                  # グリッドを点線表示
    pp.xlim([0, 0.75])                                                      # X軸最大値：0.75
    pp.ylim([0, 1])                                                         # Y軸最大値：1
    pp.xticks(np.arange(0, 0.76, 0.05))                                     # グラフX軸の目盛
    pp.yticks(np.arange(0, 1.01, 0.05))                                     # グラフY軸の目盛
    pp.plot(np.linspace(0, 1, len(mtf)), mtf, **kwargs)                     # グラフプロット(MTFデータと)
    pp.axvline(x=0.5, linestyle=":", linewidth=0.1, color="red")            # x=0.5の位置に垂直線描画(赤い点線)
    pp.axhline(y=0.5, linestyle=":", linewidth=0.1, color="red")            # y=0.5の位置に水平線描画(赤い点線)
    pp.text(0.505, 0.75, "Nyquist limit", color="red", rotation="vertical") # x=0.5の位置にナイキスト周波数の旨記述
    pp.text(0.650, 0.51, "MTF50", color="red")                              # y=0.5の位置にMTFが0.5である旨記述
    kwargs["linestyle"] = "--"  # linestyleだけ直線に変更
    pp.xlabel("cycles/pixel")   # x軸のラベル：空間周波数
    pp.ylabel("MTF")            # y軸のラベル：MTF
    pp.legend()                 # 凡例表示(凡例の数値はplot_mtf呼び出すときのに指定してある。kwargsに入ってる)

# LSFグラフの描画
def plot_lsf(images, curves, titles, suptitle):     # 画像データ, グラフデータ, グラフのタイトル, 画像のタイトル(全部リストで指定)
    if DEBUG:   # デバッグ時に実行される
        ncols = len(curves) + len(images)   # カラム数設定
        fig, axes = pp.subplots(num="curves", nrows=1, ncols=ncols, squeeze=False, clear=True, figsize=(17,9), dpi=110) # curvesの名でfig用意
        fig.canvas.set_window_title("slanted-edge-mtf: {} ESF & LSF curves".format(suptitle))   # ウィンドウ名指定
        axes = axes.flatten()   # 多次元配列を一元化
        for i, img in enumerate(images):    # それぞれの画像に対して
            axes[i].imshow(img)             # 対応する軸にimshow
            axes[i].axvline(img.shape[1] / 2, color="red", linewidth=0.7)   # それぞれのグラフの中央に垂直線プロット
            axes[i].set_title(suptitle)     # 画像のタイトルセット
        axes = axes[len(images):]           # グラフ描画のための軸準備
        for ax, curve, title in zip(axes, curves, titles):              # それぞれのグラフに対して
            ax.grid(which="both", linestyle=":")                        # グリッド表示(縦横、点線)
            ax.plot(curve * 255)                                        # カーブを255倍してプロット(元々0--1で指定？)
            ax.axvline(curve.shape[0] / 2, color="red", linewidth=0.7)  # グラフ中央に垂直線描画
            ax.set_title(title)                                         # タイトル
            ax.set_xlabel("pixel")                                      # X軸：ピクセル
            ax.set_ylabel("DN")                                         # Y軸：DN(Digital Number?)
        pp.tight_layout()       # グラフ自動成型
        pp.show(block=False)    # グラフ描画(block=Falseで処理は先に進められる)

# エッジ描画
def plot_edge(images, edge_coeffs=None, suptitle=None):
    # plots the given list of images on separate subplots, then optionally overlays each
    # subplot with a red line representing the given linear edge equation (y = ax + b)
    if DEBUG:   # デバッグ時に実行
        ncols = len(images)
        roih, roiw = images[0].shape
        fig, axes = pp.subplots(num="edges", nrows=1, ncols=ncols, sharey=True, squeeze=False, clear=True, figsize=(17,9), dpi=110)
        fig.canvas.set_window_title("slanted-edge-mtf: {} edge detection".format(suptitle))
        axes = np.array(fig.axes)
        axes = axes.flatten()
        for ax, img in zip(axes, images):
            ax.imshow(img, cmap="gray")
            ax.xaxis.tick_top()
            if edge_coeffs is not None:
                p = np.poly1d(edge_coeffs)
                xp = np.linspace(0, roiw, roiw * 4)
                yp = p(xp)
                inside = (0 <= yp) & (yp < roih)
                xp_roi = xp[inside]
                yp_roi = yp[inside]
                ax.plot(xp_roi, yp_roi, color="red", scalex=False, scaley=False)
        pp.tight_layout()
        pp.show(block=False)


def prompt(message):    # デバッグ時に引数のメッセージ表示する簡単な関数
    if DEBUG:
        input(message)  # input関数：文字表示しつつ、エンターキーで先に進むのを1行で済ませている


def enforce(expression, message_if_false, run_if_false=None):
    if not expression:              # 本来起こってほしいもの(expression)が起こらない(False)場合
        print(message_if_false)     # エラー文表示
        if run_if_false is not None:# エラー起こった時に呼ぶ関数(引数で指定されていれば)を呼ぶ
            run_if_false()
        prompt("Processing failed. Press Enter to quit...") # 一個うえで定義されてる関数。DEBUG時はこの文表示して閉じる
        sys.exit(1)                 # プログラム異常終了

# ROI選択用クラス
class ROI_selector(object):

    def __init__(self, filename):   # コンストラクタ：画像読み込みだけ
        self.image = imread(filename, verbose=False)

    def run(self, corner):          # run：ROI選択本体
        self.fig, self.ax = pp.subplots(num="selector", figsize=(17,9), dpi=110)    # num:識別番号, figsize:画像サイズ(縦横比っぽい), dpi:1インチあたりのピクセル数
        self.fig.canvas.set_window_title("slanted-edge-mtf: Edge Region Selector")
        self.ax.imshow(self.image, cmap="gray")
        """
        RectangleSelector   https://matplotlib.org/stable/api/widgets_api.html#matplotlib.widgets.RectangleSelector
        第一引数：   軸
        第二引数：   関数。マウスホールドを解除したとき(リリースされたとき)呼び出す関数を指定。eclick, ereleaseを引数とする。
                        eclick:     マウスを押した時点のデータ
                        erelease:   マウスを離した時点のデータ
        drawtype：  描画するタイプ？
        uselit：    Trueにすると描画が高速化されるらしい
        button：    トリガーに使うボタン    https://matplotlib.org/stable/api/backend_bases_api.html#matplotlib.backend_bases.MouseButton
                        左クリック：1, 中クリック：2, 右クリック：3, BackとForwardもある(何を指してるかは分からない)
        minspanx：  最小選択領域サイズ(x方向)   これより小さいと無反応にするらしい
        minspany：  最小選択領域サイズ(y方向)
        spancoords：minspan{x|y}をデータで見るかピクセル(描画上のサイズ？)で見るか
        interactive：描画後にインタラクションするか否か。(多分Falseにすると選択した瞬間閉じる)
        """
        rs = matplotlib.widgets.RectangleSelector(self.ax,
                                                  self.box_select_callback,
                                                  drawtype="box",
                                                  useblit=True,
                                                  button=[1],
                                                  minspanx=MIN_ROI_WIDTH,
                                                  minspany=MIN_ROI_HEIGHT,
                                                  spancoords="data",
                                                  interactive=True)
        # pp(pyplot).connect：第一引数のイベントに第二引数の関数を結びつける。
        # インタラクティブなプロットが作成できる                                詳細：https://qiita.com/ceptree/items/c9239ce4e442482769b3
        pp.connect("key_press_event", self.event_exit_callback) # キーを押したとき、event_exit_callback関数を呼ぶようにする
        pp.title("Select {} edge region, then press Enter".format(corner.upper()))  # グラフのタイトル描画
        pp.show(block=True)     # block=Trueにするとウィンドウ閉じるまで先に進まなくなる    詳細：https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.show.html?highlight=show
        return list(self.roi)

    def box_select_callback(self, eclick, erelease):    # 領域選択時(マウスホールド後)に呼ばれる関数
        x1, y1 = eclick.xdata, eclick.ydata             # 始点の座標を格納
        x2, y2 = erelease.xdata, erelease.ydata         # 終点の座標を格納
        self.roi = np.array([y1, y2, x1, x2]).round().astype(np.uint32) # ここで32bitのintに変換

    def event_exit_callback(self, event):
        if event.key in ["enter", "esc"]:   # 押されたキーがEnter, Escのどっちか
            pp.close("selector")            # グラフを閉じる
            # show関数でblock=Trueにしてるので、ここで閉じることで次の処理へ進めることができる

# jsonファイルの読み込み
def load_json(filename):
    with open(filename, "r") as f:  # ファイル開く
        config = json.load(f)       # (デフォルトでついてるライブラリ)jsonを利用
    return config

# jsonファイルの書き込み
def save_json(filename, config):
    with open(filename, "w") as f:  # ファイル開く
        config_str = pprint.pformat(config, indent=2, width=120)    # データを成型してる。インデント幅2, 1行最大120字
        config_str = config_str.replace('\'', '"')  # python dict => json
        f.write(config_str)         # 書き込み自体は平文。特別なライブラリは使っていない

# 設定ファイルの読み込み
def load_config(json_file):
    if json_file is not None:           # jsonファイル(ロード元のパス)が指定されている
        enforce(os.path.exists(json_file), "Config file {} does not exist.".format(json_file))  # ファイルが見つからない場合：エラー出して終了
        print("Loading configuration from {}.".format(json_file))
        config = load_json(json_file)   # config変数にjsonファイルをロード
    else:
        print("JSON config file not specified (see --help), reverting to interactive mode.")    # loadがついてない
        config = DEFAULT_CONFIG
    return config   # configを返す

# 設定ファイルの書き込み
def save_config(json_file, config):
    if json_file is not None:                                       # jsonファイル(出力先のパス)が指定されている
        print("Saving current config to {}.".format(json_file))
        save_json(json_file, config)                                # jsonファイル書き込み

########################################

def mkfilelist(folderpath, ext):
    # 指定フォルダ内の特定の拡張子のリストを作成して返す
    files = os.listdir(folderpath)
    res = []
    res2 = []
    for f in files:
        fbase, fext = os.path.splitext(f)
        if fext=="."+ext:
            res.append(folderpath + "/" + fbase + fext)
            res2.append(fbase)
    return res, res2

# ディレクトリ作成
def mkdir(directoryPath):
    if not os.path.exists(directoryPath):
        os.makedirs(directoryPath)
    return directoryPath

# 出力先が存在するかチェック
def outputFileCheck(filePath):
    return os.path.exists(filePath)

########################################

def main():
    global DEBUG                        # grobal XXX: グローバル変数の宣言
    DEBUG = argv.exists("--debug")      # オプションに--debugがあればDEBUG変数をTrueに切り替え。デバッグ用の動作に入る。便利そう
    quiet = argv.exists("--quiet")      # quietをTrueに変更(あれば)。quietは最後の「Press Enter to quit」にだけ使ってる
    json_in = argv.stringval("--load", default=None)    # --loadがあるとき、直後の文字列を格納(ロード元のjsonファイル)
    json_out = argv.stringval("--save", default=None)   # --loadがあるとき、直後の文字列を格納(セーブ先のjsonファイル)
    corners = ["center", "top-left", "top-right", "bottom-left", "bottom-right"]
    roi = argv.stringval("--roi", default="center", accepted=corners+["all"])   # --roiの直後を格納(あれば)。acceptedは選択肢っぽい
    showHelp = argv.exists("--help")
    argv.exitIfAnyUnparsedOptions()     # 例外処理的な。不要なオプションある場合に動く？
    if showHelp or len(sys.argv) < 2:   # --help使用か引数不足なとき：使い方表示してプログラム終了
        print("Usage: slanted-edge-mtf.py [options] image.{ppm|png|jpg}")
        print()
        print("  options:")
        print("    --load config.json              load configuration from JSON file")
        print("    --save config.json              save current config to JSON file")
        print("    --roi all|center|top-left|...   region to analyze; default = center")
        print("    --quiet                         silent mode, do not show any graphs")
        print("    --debug                         plot extra graphs for diagnostics")
        print("    --help                          show this help message")
        print()
        print("  interactive mode:")
        print("    mouse left + move               select region containing a slanted edge")
        print("    enter/esc                       confirm selected region, start processing")
        print()
        sys.exit(-1)

    ########################################
    filepathlist, filelist = mkfilelist(sys.argv[1], "tif") # (改変して自作関数挿入)第一引数のフォルダ名からtif形式のファイルリストを作成
    outputDir = mkdir(sys.argv[1])                          # 自作関数：格納先フォルダが無ければ作る
    
    for i in range(len(filelist)):                          # ファイルリスト一つずつ取って処理
        
        filename = filepathlist[i]
        enforce(os.path.exists(filename), "Image file {} does not exist.".format(filename)) # ファイル無い場合：エラー吐いて終了

        config = load_config(json_in)       # --loadがある場合設定のjsonファイル読み込み

        selected_rois = corners if roi == "all" else [roi]  # roiが指定されているならそれを使用。無ければcorners変数を適用
        ignored_rois = set(corners) - set(selected_rois)    # ignored_roi: roi以外の領域
        for corner in ignored_rois:
            key = "roi-{}".format(corner)  # 'top-left' => 'roi-top-left'
            config[key] = []

        if json_in is None:                     # --loadが無い場合はGUI使ってROI選択
            selector = ROI_selector(filename)   # ここでROI選択
            for roi_name in selected_rois:
                key = "roi-{}".format(roi_name)  # 'top-left' => 'roi-top-left'
                config[key] = selector.run(roi_name)

        print("=" * 40, os.path.basename(filename), "=" * 40)           # (自分で追加)処理中のファイル名表示
        results = [MTFResults(roi_name) for roi_name in selected_rois]  # MTF解析結果格納用クラスのリスト
        success = mtf(config, results, filename, outputDir)             # MTF解析(+プロット)。成功すればTrueが返る
        print("Success." if success else "Failed.")                 # 成功したか否か
        for res in results:
            res.report()    # テキストで簡単な解析結果のまとめを出力

        if DEBUG or not quiet:
            # input("Press Enter to quit...")   # 本来はquiet入れてる場合、最後にEnterを押すことで閉じることができる。コメントアウトしたので入れなくても良い
            pass

        pp.close("all") # グラフ閉じる

        save_config(json_out, config)   # (--saveあれば)jsonファイルに設定書き出し

    sys.exit(0 if success else 1)
    
    """
    filename = sys.argv[1]
    enforce(os.path.exists(filename), "Image file {} does not exist.".format(filename))

    config = load_config(json_in)

    selected_rois = corners if roi == "all" else [roi]
    ignored_rois = set(corners) - set(selected_rois)
    for corner in ignored_rois:
        key = "roi-{}".format(corner)  # 'top-left' => 'roi-top-left'
        config[key] = []

    if json_in is None:
        selector = ROI_selector(filename)
        for roi_name in selected_rois:
            key = "roi-{}".format(roi_name)  # 'top-left' => 'roi-top-left'
            config[key] = selector.run(roi_name)

    print("=" * 40, os.path.basename(filename), "=" * 40)
    results = [MTFResults(roi_name) for roi_name in selected_rois]
    success = mtf(config, results, filename)
    print("Success." if success else "Failed.")
    for res in results:
        res.report()

    if DEBUG or not quiet:
        # input("Press Enter to quit...")
        pass

    pp.close("all")

    save_config(json_out, config)

    sys.exit(0 if success else 1)
    """


if __name__ == "__main__":
    main()
