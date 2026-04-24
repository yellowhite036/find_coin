"""
台幣硬幣辨識系統 - 純 OpenCV 實作
支援：1元、5元、10元、50元
輸出：標註影片（bounding circle + 幣值標籤 + 數量統計）
"""

import cv2
import numpy as np
import argparse
import os
import sys
from collections import defaultdict


# ── 台幣硬幣特徵 ─────────────────────────────────────────────────────────────
# 依據真實台幣直徑比例定義（相對比例，不依賴實際mm）
# 1元：20mm  5元：22mm  10元：26mm  50元：28mm
COIN_PROFILES = {
    1:  {"color": (180, 180, 180), "label": "1元",  "bgr": (200, 200, 200), "ring": False},
    5:  {"color": (180, 150,  50), "label": "5元",  "bgr": ( 50, 180, 220), "ring": False},
    10: {"color": (180, 150,  50), "label": "10元", "bgr": ( 40, 140, 255), "ring": True },  # 黃銅+白鋼 雙色
    50: {"color": (180, 180, 180), "label": "50元", "bgr": (255, 180,  50), "ring": False},
}

# 畫面UI顏色
UI_BG      = (20, 20, 20)
UI_TEXT    = (240, 240, 240)
UI_ACCENT  = (80, 200, 120)
UI_WARN    = (50, 160, 255)


# ── 硬幣辨識器 ────────────────────────────────────────────────────────────────
class TWDCoinDetector:
    def __init__(self, debug: bool = False):
        self.debug = debug
        # Hough 參數（可依影片解析度調整）
        self.dp          = 1.2
        self.min_dist    = 40    # 硬幣圓心最小距離 (px)
        self.param1      = 60    # Canny 高閾值
        self.param2      = 28    # 圓心累加閾值（越小越多偽陽）
        self.min_radius  = 18    # 最小半徑 (px)
        self.max_radius  = 120   # 最大半徑 (px)

        # 顏色閾值 (HSV)
        # 銀白色：低飽和
        self.silver_sat_max  = 60
        # 黃銅色：Hue 15-35, Sat > 60
        self.brass_hue_lo    = np.array([ 8, 60,  80])
        self.brass_hue_hi    = np.array([35, 255, 255])

        # 追蹤：上一幀偵測到的硬幣（穩定標籤用）
        self._prev_coins: list[dict] = []

    # ── 前處理 ────────────────────────────────────────────────────────────────
    def _preprocess(self, frame: np.ndarray):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        # CLAHE 提升對比（解決反光問題）
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)
        return enhanced

    # ── Hough 圓偵測 ──────────────────────────────────────────────────────────
    def _detect_circles(self, processed: np.ndarray) -> np.ndarray | None:
        circles = cv2.HoughCircles(
            processed,
            cv2.HOUGH_GRADIENT,
            dp=self.dp,
            minDist=self.min_dist,
            param1=self.param1,
            param2=self.param2,
            minRadius=self.min_radius,
            maxRadius=self.max_radius,
        )
        return circles

    # ── 顏色分析：判斷硬幣材質 ────────────────────────────────────────────────
    def _analyze_color(self, frame: np.ndarray, cx: int, cy: int, r: int) -> dict:
        """
        回傳 {'is_brass': bool, 'has_ring': bool, 'mean_sat': float}
        """
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (cx, cy), max(r - 4, 5), 255, -1)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        roi_hsv = hsv[mask > 0]

        if len(roi_hsv) == 0:
            return {"is_brass": False, "has_ring": False, "mean_sat": 0}

        mean_sat = float(np.mean(roi_hsv[:, 1]))
        mean_hue = float(np.mean(roi_hsv[:, 0]))

        is_brass = (8 <= mean_hue <= 35) and (mean_sat > 55)

        # 10元：中央銀白 + 外環黃銅（分環形區分析）
        # 外環 ROI
        outer_mask = np.zeros_like(mask)
        cv2.circle(outer_mask, (cx, cy), r, 255, -1)
        cv2.circle(outer_mask, (cx, cy), max(int(r * 0.55), 5), 0, -1)
        inner_mask = np.zeros_like(mask)
        cv2.circle(inner_mask, (cx, cy), max(int(r * 0.50), 5), 255, -1)

        outer_hsv = hsv[outer_mask > 0]
        inner_hsv = hsv[inner_mask > 0]

        has_ring = False
        if len(outer_hsv) > 20 and len(inner_hsv) > 20:
            outer_sat = float(np.mean(outer_hsv[:, 1]))
            inner_sat = float(np.mean(inner_hsv[:, 1]))
            outer_hue = float(np.mean(outer_hsv[:, 0]))
            inner_hue = float(np.mean(inner_hsv[:, 0]))
            # 外環偏黃、內環偏銀 → 判定為10元
            if outer_sat > 60 and inner_sat < 55 and abs(outer_hue - inner_hue) > 8:
                has_ring = True

        return {"is_brass": is_brass, "has_ring": has_ring, "mean_sat": mean_sat}

    # ── 根據半徑比例 + 顏色分類幣值 ──────────────────────────────────────────
    def _classify(self, r: int, color_info: dict, ref_r: float) -> int:
        """
        ref_r: 本幀最大硬幣半徑（用於正規化比例）
        台幣直徑比例：1元:5元:10元:50元 ≈ 1.00 : 1.10 : 1.30 : 1.40
        """
        ratio = r / ref_r if ref_r > 0 else 1.0
        is_brass = color_info["is_brass"]
        has_ring = color_info["has_ring"]

        # 10元優先（雙色環是強特徵）
        if has_ring:
            return 10

        # 50元：最大、銀色
        if ratio >= 1.25 and not is_brass:
            return 50

        # 10元：大且黃銅（無環偵測到的備援）
        if ratio >= 1.15 and is_brass:
            return 10

        # 5元：中等、黃銅
        if ratio >= 0.90 and is_brass:
            return 5

        # 1元：最小、銀色（或任何小圓）
        return 1

    # ── 時間穩定濾波（避免閃爍）────────────────────────────────────────────
    def _stabilize(self, coins: list[dict]) -> list[dict]:
        for coin in coins:
            best_match = None
            best_dist = 30  # px 門檻
            for prev in self._prev_coins:
                d = np.hypot(coin["cx"] - prev["cx"], coin["cy"] - prev["cy"])
                if d < best_dist:
                    best_dist = d
                    best_match = prev
            if best_match is not None:
                # 用前幀分類平滑（EMA）
                if best_match["value"] == coin["value"]:
                    coin["stable"] = True
                else:
                    coin["value"] = best_match["value"]  # 保持前幀標籤
                    coin["stable"] = False
            else:
                coin["stable"] = True
        self._prev_coins = coins
        return coins

    # ── 主偵測函式（單幀）────────────────────────────────────────────────────
    def detect(self, frame: np.ndarray) -> list[dict]:
        processed = self._preprocess(frame)
        circles = self._detect_circles(processed)

        if circles is None:
            self._prev_coins = []
            return []

        circles = np.round(circles[0]).astype(int)
        ref_r = float(max(c[2] for c in circles))

        coins = []
        for (cx, cy, r) in circles:
            # 邊界保護
            if cx - r < 0 or cy - r < 0 or cx + r >= frame.shape[1] or cy + r >= frame.shape[0]:
                continue
            color_info = self._analyze_color(frame, cx, cy, r)
            value = self._classify(r, color_info, ref_r)
            coins.append({
                "cx": int(cx), "cy": int(cy), "r": int(r),
                "value": value, "stable": True,
                "is_brass": color_info["is_brass"],
                "has_ring": color_info["has_ring"],
            })

        return self._stabilize(coins)


# ── 繪製函式 ──────────────────────────────────────────────────────────────────
def draw_coins(frame: np.ndarray, coins: list[dict]) -> np.ndarray:
    out = frame.copy()

    for coin in coins:
        cx, cy, r, val = coin["cx"], coin["cy"], coin["r"], coin["value"]
        profile = COIN_PROFILES.get(val, COIN_PROFILES[1])
        color = profile["bgr"]
        label = profile["label"]

        # 外圈
        thickness = 3 if coin["stable"] else 1
        cv2.circle(out, (cx, cy), r + 4, color, thickness)
        # 中心點
        cv2.circle(out, (cx, cy), 3, color, -1)

        # 10元雙色提示（畫內環）
        if coin["has_ring"]:
            cv2.circle(out, (cx, cy), max(int(r * 0.55), 5), (200, 220, 100), 1)

        # 標籤背景
        font_scale = max(0.55, r / 55)
        thickness_txt = 2
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness_txt)
        tx, ty = cx - tw // 2, cy + th // 2
        cv2.rectangle(out, (tx - 4, ty - th - 4), (tx + tw + 4, ty + 6),
                      (20, 20, 20), -1)
        cv2.putText(out, label, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness_txt, cv2.LINE_AA)

    return out


def draw_hud(frame: np.ndarray, coins: list[dict], frame_idx: int, fps: float) -> np.ndarray:
    out = frame
    h, w = out.shape[:2]

    # 統計
    count_map: dict[int, int] = defaultdict(int)
    for c in coins:
        count_map[c["value"]] += 1
    total_value = sum(v * n for v, n in count_map.items())
    total_count = sum(count_map.values())

    # HUD 背景板
    panel_w, panel_h = 210, 40 + 28 * len(COIN_PROFILES) + 35
    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    panel[:] = (25, 25, 25)
    cv2.rectangle(panel, (0, 0), (panel_w - 1, panel_h - 1), (60, 60, 60), 1)

    # 標題
    cv2.putText(panel, "TWD Coin Detector", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, UI_ACCENT, 1, cv2.LINE_AA)
    cv2.line(panel, (8, 28), (panel_w - 8, 28), (60, 60, 60), 1)

    # 各幣值統計
    y = 50
    for val in sorted(COIN_PROFILES.keys()):
        n = count_map.get(val, 0)
        color = COIN_PROFILES[val]["bgr"]
        label = COIN_PROFILES[val]["label"]
        cv2.putText(panel, f"{label:>4s} x {n:2d}  ({val*n:4d})", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
        y += 28

    # 分隔線 + 合計
    cv2.line(panel, (8, y - 6), (panel_w - 8, y - 6), (60, 60, 60), 1)
    cv2.putText(panel, f"Total: {total_count} coins  ${total_value}", (8, y + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, UI_TEXT, 1, cv2.LINE_AA)

    # 貼到影格左上角
    margin = 12
    py1, py2 = margin, margin + panel_h
    px1, px2 = margin, margin + panel_w
    # 半透明混合
    roi = out[py1:py2, px1:px2]
    alpha = 0.80
    blended = cv2.addWeighted(panel, alpha, roi, 1 - alpha, 0)
    out[py1:py2, px1:px2] = blended

    # 右下角 Frame / Time 資訊
    time_str = f"Frame {frame_idx:05d}  {frame_idx/fps:.2f}s"
    cv2.putText(out, time_str, (w - 260, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1, cv2.LINE_AA)

    return out


# ── 主處理流程 ─────────────────────────────────────────────────────────────────
def process_video(input_path: str, output_path: str, debug: bool = False):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[ERROR] 無法開啟影片：{input_path}")
        sys.exit(1)

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[INFO] 輸入：{input_path}")
    print(f"[INFO] 解析度：{width}x{height}  FPS：{fps:.1f}  總幀數：{total}")
    print(f"[INFO] 輸出：{output_path}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    detector = TWDCoinDetector(debug=debug)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        coins = detector.detect(frame)
        annotated = draw_coins(frame, coins)
        annotated = draw_hud(annotated, coins, frame_idx, fps)

        writer.write(annotated)

        if frame_idx % 30 == 0:
            pct = frame_idx / total * 100 if total > 0 else 0
            coin_sum = sum(c["value"] for c in coins)
            print(f"  幀 {frame_idx:5d}/{total}  ({pct:5.1f}%)  "
                  f"偵測到 {len(coins)} 枚  總值 ${coin_sum}")

        frame_idx += 1

    cap.release()
    writer.release()
    print(f"\n[DONE] 已輸出 {frame_idx} 幀 → {output_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="台幣硬幣辨識 - 純OpenCV")
    parser.add_argument("input",  help="輸入影片路徑（如 coin.mp4）")
    parser.add_argument("-o", "--output", default="output.mp4", help="輸出影片路徑")
    parser.add_argument("--debug", action="store_true", help="除錯模式（輸出前處理資訊）")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] 找不到輸入檔案：{args.input}")
        sys.exit(1)

    process_video(args.input, args.output, debug=args.debug)