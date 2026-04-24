"""
台幣硬幣偵測 - 簡化版
只做圓形偵測，不分類幣值
"""

import cv2       # 匯入 OpenCV 影像處理函式庫
import numpy as np  # 匯入 NumPy 數值運算函式庫
import sys          # 匯入系統模組（用於 sys.exit 與 sys.argv）


def preprocess(frame: np.ndarray) -> np.ndarray:  # 影像前處理函式，輸入彩色影格，回傳灰階增強影像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)              # 將 BGR 彩色影像轉為灰階
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)                 # 高斯模糊降噪，核大小 9x9、標準差 2
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)) # 建立 CLAHE 物件，限制對比放大倍率 2.5
    return clahe.apply(blurred)                                  # 對模糊影像套用 CLAHE 增強對比後回傳


def detect_circles(processed: np.ndarray):  # 圓形偵測函式，輸入前處理後的灰階影像，回傳圓形陣列或 None
    return cv2.HoughCircles(
        processed,           # 輸入灰階影像
        cv2.HOUGH_GRADIENT,  # 使用梯度法進行 Hough 變換
        dp=1.0,              # 累加器解析度與影像解析度的比值
        minDist=40,          # 兩圓圓心的最小距離（px），避免同一硬幣偵測到多個圓
        param1=60,           # Canny 邊緣偵測的高閾值
        param2=80,           # 圓心累加閾值，越小偵測越多但誤判也越多
        minRadius=18,        # 偵測圓形的最小半徑（px）
        maxRadius=120,       # 偵測圓形的最大半徑（px）
    )


def draw_circles(frame: np.ndarray, circles) -> np.ndarray:  # 繪製圓形標註函式，回傳標註後的影格
    out = frame.copy()   # 複製原始影格，避免修改原始資料
    if circles is None:  # 若未偵測到任何圓形
        return out       # 直接回傳原始影格

    for (cx, cy, r) in np.round(circles[0]).astype(int):       # 遍歷每個偵測到的圓（座標取整數）
        cv2.circle(out, (cx, cy), r + 4, (0, 255, 0), 2)       # 繪製綠色外圈（比硬幣半徑略大 4px）
        cv2.circle(out, (cx, cy), 3, (0, 255, 0), -1)          # 繪製綠色圓心實心點

        coord_text = f"({cx}, {cy})"                            # 組合座標文字字串
        (tw, th), _ = cv2.getTextSize(coord_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)  # 計算文字寬高
        tx, ty = cx - tw // 2, cy - r - 10                     # 文字置中於圓形正上方
        cv2.rectangle(out, (tx - 3, ty - th - 2), (tx + tw + 3, ty + 4), (20, 20, 20), -1)  # 繪製文字背景矩形
        cv2.putText(out, coord_text, (tx, ty),                  # 繪製座標文字
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)  # 綠色、反鋸齒

    return out  # 回傳標註後的影格


def process_video(input_path: str, output_path: str):  # 影片處理主函式
    cap = cv2.VideoCapture(input_path)  # 開啟輸入影片
    if not cap.isOpened():              # 若影片無法開啟
        print(f"[ERROR] 無法開啟影片：{input_path}")  # 輸出錯誤訊息
        sys.exit(1)                     # 結束程式

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0          # 取得影片 FPS（取不到則預設 30）
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))     # 取得影片寬度
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))    # 取得影片高度
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))     # 取得影片總幀數

    print(f"[INFO] {width}x{height}  FPS:{fps:.1f}  總幀數:{total}")  # 輸出影片基本資訊

    writer = cv2.VideoWriter(               # 建立影片寫入物件
        output_path,                        # 輸出檔案路徑
        cv2.VideoWriter_fourcc(*"mp4v"),    # 設定編碼格式為 mp4v
        fps,                                # 輸出影片的 FPS
        (width, height)                     # 輸出影片的解析度
    )

    frame_idx = 0   # 初始化影格計數器
    while True:     # 持續讀取影格直到影片結束
        ret, frame = cap.read()  # 讀取下一影格
        if not ret:              # 若讀取失敗（影片結束）
            break                # 跳出迴圈

        processed = preprocess(frame)        # 對影格進行前處理
        circles   = detect_circles(processed)  # 偵測圓形
        annotated = draw_circles(frame, circles)  # 繪製標註

        writer.write(annotated)  # 將標註後的影格寫入輸出影片

        if frame_idx % 30 == 0:  # 每 30 幀輸出一次進度（約每秒一次）
            count = len(circles[0]) if circles is not None else 0  # 計算本幀偵測到的圓形數量
            pct = frame_idx / total * 100 if total > 0 else 0      # 計算處理進度百分比
            print(f"  幀 {frame_idx:5d}/{total} ({pct:5.1f}%)  偵測到 {count} 枚")  # 輸出進度資訊

        frame_idx += 1  # 影格計數器加一

    cap.release()     # 釋放影片讀取資源
    writer.release()  # 釋放影片寫入資源
    print(f"[DONE] 輸出 {frame_idx} 幀 → {output_path}")  # 輸出完成訊息


if __name__ == "__main__":  # 當此檔案作為主程式執行時
    if len(sys.argv) < 2:   # 若未提供輸入影片參數
        print("用法：python coin_simple.py <輸入影片> [輸出影片]")  # 輸出使用說明
        sys.exit(1)         # 結束程式

    input_path  = sys.argv[1]                                  # 取得輸入影片路徑（第一個參數）
    output_path = sys.argv[2] if len(sys.argv) > 2 else "output.mp4"  # 取得輸出路徑（第二個參數，預設 output.mp4）

    process_video(input_path, output_path)  # 執行影片處理主函式