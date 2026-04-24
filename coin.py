"""
台幣硬幣辨識系統 - 純 OpenCV 實作
支援：1元、5元、10元、50元
輸出：標註影片（bounding circle + 幣值標籤 + 數量統計）
"""

import cv2  # 匯入 OpenCV 影像處理函式庫
import numpy as np  # 匯入 NumPy 數值運算函式庫
import argparse  # 匯入命令列參數解析模組
import os  # 匯入作業系統相關模組
import sys  # 匯入系統模組（用於 sys.exit）
from collections import defaultdict  # 匯入 defaultdict，用於計數統計


# ── 台幣硬幣特徵 ─────────────────────────────────────────────────────────────
# 依據真實台幣直徑比例定義（相對比例，不依賴實際mm）
# 1元：20mm  5元：22mm  10元：26mm  50元：28mm
COIN_PROFILES = {  # 定義各幣值的顯示屬性字典
    1:  {"color": (180, 180, 180), "label": "1元",  "bgr": (200, 200, 200), "ring": False},  # 1元：銀灰色，無雙色環
    5:  {"color": (180, 150,  50), "label": "5元",  "bgr": ( 50, 180, 220), "ring": False},  # 5元：黃銅色，無雙色環
    10: {"color": (180, 150,  50), "label": "10元", "bgr": ( 40, 140, 255), "ring": True },  # 10元：黃銅+白鋼 雙色環
    50: {"color": (180, 180, 180), "label": "50元", "bgr": (255, 180,  50), "ring": False},  # 50元：銀色，無雙色環
}

# 畫面UI顏色
UI_BG      = (20, 20, 20)    # UI 背景色（深灰）
UI_TEXT    = (240, 240, 240)  # UI 一般文字色（白）
UI_ACCENT  = (80, 200, 120)   # UI 強調色（綠）
UI_WARN    = (50, 160, 255)   # UI 警告色（橘黃）


# ── 硬幣辨識器 ────────────────────────────────────────────────────────────────
class TWDCoinDetector:  # 定義台幣硬幣偵測器類別
    def __init__(self, debug: bool = False):  # 建構子，接受 debug 參數
        self.debug = debug  # 儲存除錯模式旗標
        # Hough 參數（可依影片解析度調整）
        self.dp          = 1.2   # Hough 累加器解析度與影像解析度的比值
        self.min_dist    = 40    # 硬幣圓心最小距離 (px)，避免同一硬幣偵測到多個圓
        self.param1      = 60    # Canny 邊緣偵測高閾值
        self.param2      = 28    # 圓心累加閾值（越小越多偽陽性）
        self.min_radius  = 18    # 最小硬幣半徑 (px)
        self.max_radius  = 120   # 最大硬幣半徑 (px)

        # 顏色閾值 (HSV)
        self.silver_sat_max  = 60  # 銀白色：飽和度上限（低飽和 = 銀色）
        # 黃銅色 HSV 範圍：Hue 15-35, Sat > 60
        self.brass_hue_lo    = np.array([ 8, 60,  80])   # 黃銅色 HSV 下界
        self.brass_hue_hi    = np.array([35, 255, 255])  # 黃銅色 HSV 上界

        # 追蹤：上一幀偵測到的硬幣（用於時間穩定濾波）
        self._prev_coins: list[dict] = []  # 初始化前幀硬幣清單為空

    # ── 前處理 ────────────────────────────────────────────────────────────────
    def _preprocess(self, frame: np.ndarray):  # 影像前處理函式
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 將彩色影像轉為灰階
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)  # 高斯模糊降噪，核大小 9x9
        # CLAHE 提升對比（解決反光問題）
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))  # 建立 CLAHE 物件，限制對比放大倍率 2.5
        enhanced = clahe.apply(blurred)  # 對模糊影像套用 CLAHE 增強對比
        return enhanced  # 回傳增強後的灰階影像

    # ── Hough 圓偵測 ──────────────────────────────────────────────────────────
    def _detect_circles(self, processed: np.ndarray) -> np.ndarray | None:  # 使用 Hough 變換偵測圓形
        circles = cv2.HoughCircles(  # 呼叫 OpenCV Hough 圓偵測
            processed,               # 輸入灰階影像
            cv2.HOUGH_GRADIENT,      # 使用梯度法
            dp=self.dp,              # 累加器解析度比值
            minDist=self.min_dist,   # 圓心最小距離
            param1=self.param1,      # Canny 高閾值
            param2=self.param2,      # 圓心累加閾值
            minRadius=self.min_radius,  # 最小半徑
            maxRadius=self.max_radius,  # 最大半徑
        )
        return circles  # 回傳偵測到的圓形陣列（若無則為 None）

    # ── 顏色分析：判斷硬幣材質 ────────────────────────────────────────────────
    def _analyze_color(self, frame: np.ndarray, cx: int, cy: int, r: int) -> dict:  # 分析硬幣顏色特徵
        """
        回傳 {'is_brass': bool, 'has_ring': bool, 'mean_sat': float}
        """
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)  # 建立與影像同尺寸的全黑遮罩
        cv2.circle(mask, (cx, cy), max(r - 4, 5), 255, -1)  # 在遮罩上繪製實心圓（略縮邊緣避免干擾）

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # 將影像從 BGR 轉換為 HSV 色彩空間
        roi_hsv = hsv[mask > 0]  # 取出遮罩範圍內的 HSV 像素值

        if len(roi_hsv) == 0:  # 若遮罩內無像素（邊界情況）
            return {"is_brass": False, "has_ring": False, "mean_sat": 0}  # 回傳預設值

        mean_sat = float(np.mean(roi_hsv[:, 1]))  # 計算飽和度平均值
        mean_hue = float(np.mean(roi_hsv[:, 0]))  # 計算色相平均值

        is_brass = (8 <= mean_hue <= 35) and (mean_sat > 55)  # 判斷是否為黃銅色（色相在黃色範圍且飽和度足夠）

        # 10元：中央銀白 + 外環黃銅（分環形區分析）
        # 外環 ROI
        outer_mask = np.zeros_like(mask)  # 建立外環遮罩（全黑）
        cv2.circle(outer_mask, (cx, cy), r, 255, -1)  # 繪製完整圓形（白色填充）
        cv2.circle(outer_mask, (cx, cy), max(int(r * 0.55), 5), 0, -1)  # 挖去內部 55% 半徑區域，剩下外環
        inner_mask = np.zeros_like(mask)  # 建立內圓遮罩（全黑）
        cv2.circle(inner_mask, (cx, cy), max(int(r * 0.50), 5), 255, -1)  # 繪製內部 50% 半徑的實心圓

        outer_hsv = hsv[outer_mask > 0]  # 取出外環區域的 HSV 像素
        inner_hsv = hsv[inner_mask > 0]  # 取出內圓區域的 HSV 像素

        has_ring = False  # 預設沒有雙色環
        if len(outer_hsv) > 20 and len(inner_hsv) > 20:  # 確保兩區域都有足夠像素
            outer_sat = float(np.mean(outer_hsv[:, 1]))  # 計算外環飽和度平均
            inner_sat = float(np.mean(inner_hsv[:, 1]))  # 計算內圓飽和度平均
            outer_hue = float(np.mean(outer_hsv[:, 0]))  # 計算外環色相平均
            inner_hue = float(np.mean(inner_hsv[:, 0]))  # 計算內圓色相平均
            # 外環偏黃、內環偏銀 → 判定為10元
            if outer_sat > 60 and inner_sat < 55 and abs(outer_hue - inner_hue) > 8:  # 外環飽和度高、內環飽和度低、色相差異明顯
                has_ring = True  # 確認具有雙色環特徵

        return {"is_brass": is_brass, "has_ring": has_ring, "mean_sat": mean_sat}  # 回傳顏色分析結果

    # ── 根據半徑比例 + 顏色分類幣值 ──────────────────────────────────────────
    def _classify(self, r: int, color_info: dict, ref_r: float) -> int:  # 根據大小與顏色判斷幣值
        """
        ref_r: 本幀最大硬幣半徑（用於正規化比例）
        台幣直徑比例：1元:5元:10元:50元 ≈ 1.00 : 1.10 : 1.30 : 1.40
        """
        ratio = r / ref_r if ref_r > 0 else 1.0  # 計算當前硬幣相對於最大硬幣的半徑比例
        is_brass = color_info["is_brass"]  # 取出黃銅色判斷結果
        has_ring = color_info["has_ring"]  # 取出雙色環判斷結果

        # 10元優先（雙色環是強特徵）
        if has_ring:  # 若偵測到雙色環
            return 10  # 直接判定為 10 元

        # 50元：最大、銀色
        if ratio >= 1.25 and not is_brass:  # 相對半徑大且為銀色
            return 50  # 判定為 50 元

        # 10元：大且黃銅（無環偵測到的備援路徑）
        if ratio >= 1.15 and is_brass:  # 相對半徑較大且為黃銅色
            return 10  # 判定為 10 元

        # 5元：中等、黃銅
        if ratio >= 0.90 and is_brass:  # 相對半徑中等且為黃銅色
            return 5  # 判定為 5 元

        # 1元：最小、銀色（或任何小圓形）
        return 1  # 其餘情況一律判定為 1 元

    # ── 時間穩定濾波（避免標籤閃爍）────────────────────────────────────────────
    def _stabilize(self, coins: list[dict]) -> list[dict]:  # 利用前幀資料穩定當前幀的辨識結果
        for coin in coins:  # 遍歷當前幀所有硬幣
            best_match = None  # 初始化最佳匹配為空
            best_dist = 30  # 設定位置匹配門檻（像素距離）
            for prev in self._prev_coins:  # 遍歷前幀所有硬幣
                d = np.hypot(coin["cx"] - prev["cx"], coin["cy"] - prev["cy"])  # 計算當前與前幀硬幣的歐氏距離
                if d < best_dist:  # 若距離小於門檻
                    best_dist = d  # 更新最小距離
                    best_match = prev  # 記錄最佳匹配的前幀硬幣
            if best_match is not None:  # 若找到對應的前幀硬幣
                if best_match["value"] == coin["value"]:  # 若幣值與前幀相同
                    coin["stable"] = True  # 標記為穩定
                else:  # 若幣值與前幀不同（可能是誤判）
                    coin["value"] = best_match["value"]  # 保持前幀的幣值（避免閃爍）
                    coin["stable"] = False  # 標記為不穩定
            else:  # 若無前幀對應（新出現的硬幣）
                coin["stable"] = True  # 視為穩定（新硬幣）
        self._prev_coins = coins  # 將當前幀硬幣存為下一幀的前幀參考
        return coins  # 回傳穩定後的硬幣清單

    # ── 主偵測函式（單幀）────────────────────────────────────────────────────
    def detect(self, frame: np.ndarray) -> list[dict]:  # 對單一影格執行完整硬幣偵測流程
        processed = self._preprocess(frame)  # 對影格進行前處理（灰階、模糊、CLAHE）
        circles = self._detect_circles(processed)  # 使用 Hough 變換偵測圓形

        if circles is None:  # 若未偵測到任何圓形
            self._prev_coins = []  # 清空前幀記錄
            return []  # 回傳空清單

        circles = np.round(circles[0]).astype(int)  # 將圓形座標取整數（移除多餘維度）
        ref_r = float(max(c[2] for c in circles))  # 取本幀最大半徑作為基準（用於比例計算）

        coins = []  # 初始化本幀硬幣清單
        for (cx, cy, r) in circles:  # 遍歷每個偵測到的圓形
            # 邊界保護：確保圓形不超出影像邊界
            if cx - r < 0 or cy - r < 0 or cx + r >= frame.shape[1] or cy + r >= frame.shape[0]:  # 若圓形超出影像範圍
                continue  # 跳過此圓形
            color_info = self._analyze_color(frame, cx, cy, r)  # 分析此圓形的顏色特徵
            value = self._classify(r, color_info, ref_r)  # 根據大小與顏色分類幣值
            coins.append({  # 將硬幣資訊加入清單
                "cx": int(cx), "cy": int(cy), "r": int(r),  # 圓心座標與半徑
                "value": value, "stable": True,  # 幣值與穩定旗標
                "is_brass": color_info["is_brass"],  # 是否為黃銅色
                "has_ring": color_info["has_ring"],  # 是否有雙色環
            })

        return self._stabilize(coins)  # 經時間穩定濾波後回傳結果


# ── 繪製函式 ──────────────────────────────────────────────────────────────────
def draw_coins(frame: np.ndarray, coins: list[dict]) -> np.ndarray:  # 在影格上繪製硬幣標註
    out = frame.copy()  # 複製原始影格，避免修改原始資料

    for coin in coins:  # 遍歷所有偵測到的硬幣
        cx, cy, r, val = coin["cx"], coin["cy"], coin["r"], coin["value"]  # 取出硬幣基本屬性
        profile = COIN_PROFILES.get(val, COIN_PROFILES[1])  # 取得對應幣值的顯示設定（找不到則用1元）
        color = profile["bgr"]  # 取出 BGR 顏色
        label = profile["label"]  # 取出幣值標籤文字

        # 外圈標註
        thickness = 3 if coin["stable"] else 1  # 穩定硬幣用粗線，不穩定用細線
        cv2.circle(out, (cx, cy), r + 4, color, thickness)  # 在硬幣外側繪製標註圓圈
        # 中心點
        cv2.circle(out, (cx, cy), 3, color, -1)  # 在圓心繪製小實心圓

        # 10元雙色提示（畫內環線）
        if coin["has_ring"]:  # 若確認為10元雙色硬幣
            cv2.circle(out, (cx, cy), max(int(r * 0.55), 5), (200, 220, 100), 1)  # 繪製內外環分界線

        # 標籤背景
        font_scale = max(0.55, r / 55)  # 依硬幣大小動態調整字體大小
        thickness_txt = 2  # 文字線條粗細
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness_txt)  # 計算文字寬高
        tx, ty = cx - tw // 2, cy + th // 2  # 計算文字置中位置
        cv2.rectangle(out, (tx - 4, ty - th - 4), (tx + tw + 4, ty + 6),  # 繪製標籤背景矩形
                      (20, 20, 20), -1)  # 深灰色填充
        cv2.putText(out, label, (tx, ty),  # 繪製幣值文字
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness_txt, cv2.LINE_AA)  # 使用反鋸齒繪製

    return out  # 回傳標註後的影格


def draw_hud(frame: np.ndarray, coins: list[dict], frame_idx: int, fps: float) -> np.ndarray:  # 繪製 HUD 統計面板
    out = frame  # 直接在傳入影格上操作（不複製）
    h, w = out.shape[:2]  # 取得影格高度與寬度

    # 統計各幣值數量
    count_map: dict[int, int] = defaultdict(int)  # 建立幣值計數字典（預設值為0）
    for c in coins:  # 遍歷所有硬幣
        count_map[c["value"]] += 1  # 對應幣值計數加一
    total_value = sum(v * n for v, n in count_map.items())  # 計算硬幣總金額
    total_count = sum(count_map.values())  # 計算硬幣總數量

    # HUD 面板尺寸
    panel_w, panel_h = 210, 40 + 28 * len(COIN_PROFILES) + 35  # 依據幣值種類數動態計算面板高度
    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)  # 建立全黑面板影像
    panel[:] = (25, 25, 25)  # 設定面板背景為深灰色
    cv2.rectangle(panel, (0, 0), (panel_w - 1, panel_h - 1), (60, 60, 60), 1)  # 繪製面板外框

    # 標題列
    cv2.putText(panel, "TWD Coin Detector", (8, 22),  # 繪製標題文字
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, UI_ACCENT, 1, cv2.LINE_AA)  # 使用強調色
    cv2.line(panel, (8, 28), (panel_w - 8, 28), (60, 60, 60), 1)  # 繪製標題下方分隔線

    # 各幣值統計列
    y = 50  # 第一行文字的 y 座標起點
    for val in sorted(COIN_PROFILES.keys()):  # 依幣值由小到大排序輸出
        n = count_map.get(val, 0)  # 取得此幣值的數量（不存在則為0）
        color = COIN_PROFILES[val]["bgr"]  # 取得對應幣值的顯示顏色
        label = COIN_PROFILES[val]["label"]  # 取得對應幣值的標籤文字
        cv2.putText(panel, f"{label:>4s} x {n:2d}  ({val*n:4d})", (10, y),  # 繪製幣值統計文字（含小計金額）
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)  # 使用對應幣值的顏色
        y += 28  # 下移一行間距

    # 分隔線與合計列
    cv2.line(panel, (8, y - 6), (panel_w - 8, y - 6), (60, 60, 60), 1)  # 繪製合計上方分隔線
    cv2.putText(panel, f"Total: {total_count} coins  ${total_value}", (8, y + 14),  # 繪製合計文字
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, UI_TEXT, 1, cv2.LINE_AA)  # 使用白色文字

    # 將面板貼到影格左上角（半透明混合）
    margin = 12  # 面板距影格邊緣的邊距
    py1, py2 = margin, margin + panel_h  # 計算面板在影格上的垂直範圍
    px1, px2 = margin, margin + panel_w  # 計算面板在影格上的水平範圍
    roi = out[py1:py2, px1:px2]  # 取出影格中對應位置的區域
    alpha = 0.80  # 設定面板不透明度（0=全透明, 1=不透明）
    blended = cv2.addWeighted(panel, alpha, roi, 1 - alpha, 0)  # 將面板與影格進行加權混合
    out[py1:py2, px1:px2] = blended  # 將混合結果寫回影格

    # 右下角顯示影格與時間資訊
    time_str = f"Frame {frame_idx:05d}  {frame_idx/fps:.2f}s"  # 組合影格編號與時間字串
    cv2.putText(out, time_str, (w - 260, h - 14),  # 在影格右下角繪製時間資訊
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1, cv2.LINE_AA)  # 使用灰色文字

    return out  # 回傳繪製完成的影格


# ── 主處理流程 ─────────────────────────────────────────────────────────────────
def process_video(input_path: str, output_path: str, debug: bool = False):  # 影片處理主函式
    cap = cv2.VideoCapture(input_path)  # 開啟輸入影片
    if not cap.isOpened():  # 若影片無法開啟
        print(f"[ERROR] 無法開啟影片：{input_path}")  # 輸出錯誤訊息
        sys.exit(1)  # 結束程式

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0  # 取得影片 FPS（若取不到則預設30）
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # 取得影片寬度
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 取得影片高度
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))   # 取得影片總幀數

    print(f"[INFO] 輸入：{input_path}")  # 輸出輸入路徑資訊
    print(f"[INFO] 解析度：{width}x{height}  FPS：{fps:.1f}  總幀數：{total}")  # 輸出影片基本資訊
    print(f"[INFO] 輸出：{output_path}")  # 輸出輸出路徑資訊

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 設定輸出影片編碼格式為 mp4v
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))  # 建立影片寫入物件

    detector = TWDCoinDetector(debug=debug)  # 建立硬幣偵測器實例
    frame_idx = 0  # 初始化影格計數器

    while True:  # 持續讀取影格直到影片結束
        ret, frame = cap.read()  # 讀取下一影格
        if not ret:  # 若讀取失敗（影片結束）
            break  # 跳出迴圈

        coins = detector.detect(frame)  # 對當前影格執行硬幣偵測
        annotated = draw_coins(frame, coins)  # 在影格上繪製硬幣標註
        annotated = draw_hud(annotated, coins, frame_idx, fps)  # 在影格上繪製 HUD 面板

        writer.write(annotated)  # 將標註後的影格寫入輸出影片

        if frame_idx % 30 == 0:  # 每30幀輸出一次進度（約每秒一次）
            pct = frame_idx / total * 100 if total > 0 else 0  # 計算處理進度百分比
            coin_sum = sum(c["value"] for c in coins)  # 計算當前幀偵測到的硬幣總值
            print(f"  幀 {frame_idx:5d}/{total}  ({pct:5.1f}%)  "  # 輸出進度與偵測結果
                  f"偵測到 {len(coins)} 枚  總值 ${coin_sum}")  # 輸出硬幣數量與總值

        frame_idx += 1  # 影格計數器加一

    cap.release()     # 釋放影片讀取資源
    writer.release()  # 釋放影片寫入資源
    print(f"\n[DONE] 已輸出 {frame_idx} 幀 → {output_path}")  # 輸出完成訊息


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":  # 當此檔案作為主程式執行時
    parser = argparse.ArgumentParser(description="台幣硬幣辨識 - 純OpenCV")  # 建立命令列參數解析器
    parser.add_argument("input",  help="輸入影片路徑（如 coin.mp4）")  # 定義必填的輸入影片參數
    parser.add_argument("-o", "--output", default="output.mp4", help="輸出影片路徑")  # 定義輸出路徑參數（預設 output.mp4）
    parser.add_argument("--debug", action="store_true", help="除錯模式（輸出前處理資訊）")  # 定義除錯模式旗標參數
    args = parser.parse_args()  # 解析命令列參數

    if not os.path.exists(args.input):  # 若輸入檔案不存在
        print(f"[ERROR] 找不到輸入檔案：{args.input}")  # 輸出錯誤訊息
        sys.exit(1)  # 結束程式

    process_video(args.input, args.output, debug=args.debug)  # 執行影片處理主函式