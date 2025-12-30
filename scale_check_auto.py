import re
import math
import statistics
from dataclasses import dataclass
from typing import List, Optional, Tuple

import fitz  # pymupdf
import numpy as np
from PIL import Image

# opencv + easyocr (you must install)
import cv2
import easyocr


@dataclass
class CheckResult:
    found: bool
    pass_ok: bool
    implied_ratio: Optional[float] = None
    note: str = ""


def extract_given_scale_ratio(pdf_path: str) -> Optional[float]:
    """
    Extracts given scale like 'Scale: 1 : 100' from PDF text.
    Returns 100.0 for 1:100
    """
    doc = fitz.open(pdf_path)
    text = ""
    for i in range(min(3, doc.page_count)):
        text += "\n" + doc[i].get_text("text")

    # Common patterns: "Scale: 1 : 100" or "Scale 1 : 100" or "1 : 100"
    m = re.search(r"Scale\s*:\s*1\s*:\s*([0-9]+(?:\.[0-9]+)?)", text, re.IGNORECASE)
    if not m:
        # fallback
        m = re.search(r"\b1\s*:\s*([0-9]+(?:\.[0-9]+)?)\b", text)
    if not m:
        return None
    return float(m.group(1))


def render_pdf_page(pdf_path: str, page_index: int = 0, dpi: int = 300) -> np.ndarray:
    """
    Render a PDF page to an OpenCV BGR image.
    """
    doc = fitz.open(pdf_path)
    page = doc[page_index]
    pix = page.get_pixmap(dpi=dpi, alpha=False)
    img = Image.open(fitz.open("png", pix.tobytes("png")))
    rgb = np.array(img)  # RGB
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr


def mm_per_pixel(dpi: int) -> float:
    return 25.4 / dpi


def ratio_close(implied: float, given: float, tol_pct: float = 2.0) -> bool:
    if given == 0:
        return False
    diff_pct = abs(implied - given) / given * 100.0
    return diff_pct <= tol_pct


def detect_scale_bar_and_ratio(
    bgr: np.ndarray,
    dpi: int,
    given_ratio: float,
    tol_pct: float = 2.0
) -> CheckResult:
    """
    Heuristic scale-bar detection:
    - Looks for long rectangular bar-like shapes near the title block / bottom area.
    - OCR nearby for labels like 'm' and numbers.
    If found, compute implied ratio.
    NOTE: Many drawings do not have a scale bar -> will return found=False.
    """
    h, w = bgr.shape[:2]
    # Focus bottom 25% where scale bars often live
    roi = bgr[int(h * 0.75):h, 0:w].copy()

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours for long thin rectangles
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        area = cw * ch
        # long & thin bar-ish
        if cw > 300 and ch < 60 and area > 2000:
            candidates.append((x, y, cw, ch))

    if not candidates:
        return CheckResult(found=False, pass_ok=False, note="Scale bar shape not found (common).")

    # OCR on ROI to find any "m" and numbers
    reader = easyocr.Reader(["en"], gpu=False)
    ocr = reader.readtext(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB), detail=1)

    # Try to locate a candidate that has nearby OCR with 'm' or 'mm'
    best = None
    best_score = 0

    for (x, y, cw, ch) in candidates:
        cx = x + cw / 2
        cy = y + ch / 2
        score = 0
        texts = []
        for bbox, txt, conf in ocr:
            # bbox is 4 points
            bx = sum(p[0] for p in bbox) / 4
            by = sum(p[1] for p in bbox) / 4
            # near the bar
            if abs(bx - cx) < 600 and abs(by - cy) < 200:
                t = txt.strip().lower()
                texts.append(t)
                if "m" == t or t.endswith("m") or "mm" in t:
                    score += 2
                if re.fullmatch(r"[0-9]+(\.[0-9]+)?", t):
                    score += 1
        if score > best_score:
            best_score = score
            best = (x, y, cw, ch, texts)

    if not best or best_score < 2:
        return CheckResult(found=False, pass_ok=False, note="Scale bar candidates exist, but labels not reliable.")

    x, y, cw, ch, texts = best

    # Try to infer real length from OCR texts (very heuristic)
    # Example expected: "0 1 2 5 m" -> take max number as meters
    nums = []
    unit_m = False
    for t in texts:
        if t.endswith("m") or t == "m":
            unit_m = True
        m = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)", t)
        if m:
            nums.append(float(m.group(1)))

    if not nums or not unit_m:
        return CheckResult(found=True, pass_ok=False, note="Scale bar found, but could not read numeric meters.")

    meters = max(nums)
    real_mm = meters * 1000.0

    paper_mm = cw * mm_per_pixel(dpi)
    implied = real_mm / paper_mm if paper_mm > 0 else None

    if implied is None:
        return CheckResult(found=True, pass_ok=False, note="Scale bar measurement failed.")

    ok = ratio_close(implied, given_ratio, tol_pct=tol_pct)
    note = f"bar_width_px={cw}, meters≈{meters:g}, paper_mm≈{paper_mm:.2f}"
    return CheckResult(found=True, pass_ok=ok, implied_ratio=implied, note=note)


def detect_dimension_ratios_auto(
    bgr: np.ndarray,
    dpi: int,
    given_ratio: float,
    tol_pct: float = 2.0,
    top_n: int = 10
) -> CheckResult:
    """
    Auto known-dimension check:
    - OCR numbers (dimension labels)
    - For each label, find nearest long line segment (Hough) and use that as paper length
    - Compute multiple implied ratios; use median as final
    """
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    reader = easyocr.Reader(["en"], gpu=False)
    ocr = reader.readtext(rgb, detail=1)

    # Keep only likely dimension numbers (mm): usually 3 to 6 digits (e.g., 29680, 37522)
    dim_candidates = []
    for bbox, txt, conf in ocr:
        t = txt.strip()
        if re.fullmatch(r"[0-9]{3,6}", t):
            val = int(t)
            # avoid tiny numbers like 100, 200 that appear everywhere
            if val >= 1000:
                # center of bbox
                cx = sum(p[0] for p in bbox) / 4
                cy = sum(p[1] for p in bbox) / 4
                dim_candidates.append((val, cx, cy, conf))

    if not dim_candidates:
        return CheckResult(found=False, pass_ok=False, note="No dimension numbers detected by OCR.")

    # Detect line segments in whole drawing
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # Hough lines
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=120,
        minLineLength=250,
        maxLineGap=20
    )

    if lines is None:
        return CheckResult(found=False, pass_ok=False, note="No line segments detected for dimension lines.")

    # Prepare lines list with length and orientation
    line_list = []
    for (x1, y1, x2, y2) in lines[:, 0]:
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length < 250:
            continue
        angle = abs(math.degrees(math.atan2(dy, dx)))
        # keep mostly horizontal or vertical lines
        if angle < 10 or abs(angle - 90) < 10:
            mx = (x1 + x2) / 2
            my = (y1 + y2) / 2
            line_list.append((x1, y1, x2, y2, length, mx, my, angle))

    if not line_list:
        return CheckResult(found=False, pass_ok=False, note="No usable horizontal/vertical lines found.")

    # For each dimension label, find nearest long line and compute implied ratio
    ratios = []
    mmpp = mm_per_pixel(dpi)

    for val, cx, cy, conf in sorted(dim_candidates, key=lambda x: -x[3])[:200]:
        # search nearest line within a window
        best = None
        best_score = 1e18

        for (x1, y1, x2, y2, length, mx, my, angle) in line_list:
            # distance from label to line midpoint
            d = math.hypot(mx - cx, my - cy)
            # must be reasonably near
            if d > 500:
                continue
            # prefer longer lines and nearer distance
            score = d - 0.02 * length
            if score < best_score:
                best_score = score
                best = (length, d)

        if best:
            length_px, dist = best
            paper_mm = length_px * mmpp
            if paper_mm <= 0:
                continue
            implied = val / paper_mm
            # ignore insane results
            if 10 <= implied <= 500:
                ratios.append(implied)

    if len(ratios) < 3:
        return CheckResult(found=False, pass_ok=False, note="Not enough reliable dimension-line matches.")

    # Take median of top_n closest to given ratio (reduces noise)
    ratios_sorted = sorted(ratios, key=lambda r: abs(r - given_ratio))
    take = ratios_sorted[:max(3, min(top_n, len(ratios_sorted)))]
    implied_med = statistics.median(take)

    ok = ratio_close(implied_med, given_ratio, tol_pct=tol_pct)
    note = f"ratios_used={len(take)}/{len(ratios)} median≈{implied_med:.2f}"
    return CheckResult(found=True, pass_ok=ok, implied_ratio=implied_med, note=note)


def final_decision(scale_bar: CheckResult, known_dim: CheckResult) -> Tuple[bool, str]:
    """
    Your rule:
    ✔ First validate using scale bar
    ✔ Cross-check using known dimensions
    ✔ If both agree → scale is correct
    ✔ If mismatch → invalid
    plus practical fallback:
    - if only one is available, use that (OR) but mention confidence
    """
    if scale_bar.found and known_dim.found:
        if scale_bar.pass_ok and known_dim.pass_ok:
            return True, "Both checks PASS and agree."
        if scale_bar.pass_ok != known_dim.pass_ok:
            return False, "Mismatch between scale bar and known dimension checks."
        return False, "Both checks found but both FAIL."
    if scale_bar.found:
        return scale_bar.pass_ok, "Only scale bar check available (known-dimension not reliable)."
    if known_dim.found:
        return known_dim.pass_ok, "Only known-dimension check available (scale bar not found)."
    return False, "No reliable check was available."


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", help="Path to drawing PDF")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--tol", type=float, default=2.0, help="tolerance percent (default 2.0%)")
    args = ap.parse_args()

    pdf_path = args.pdf
    dpi = args.dpi
    tol = args.tol

    given_scale_ratio = extract_given_scale_ratio(pdf_path)
    if given_scale_ratio is None:
        print("[FAIL] Could not detect given_scale_ratio from PDF text.")
        return

    print(f"[OK] given_scale_ratio = {given_scale_ratio:.2f}  (example: 100.00 means 1:100)")

    bgr = render_pdf_page(pdf_path, page_index=0, dpi=dpi)

    print("\n[Scale bar check]")
    sb = detect_scale_bar_and_ratio(bgr, dpi=dpi, given_ratio=given_scale_ratio, tol_pct=tol)
    print(f"  Found: {'YES' if sb.found else 'NO'}")
    if sb.implied_ratio is not None:
        print(f"  Implied ratio: {sb.implied_ratio:.2f}")
        print(f"  Compare with given: {given_scale_ratio:.2f}")
        print(f"  Result: {'PASS' if sb.pass_ok else 'FAIL'} (tol {tol:.1f}%)")
    print(f"  Note: {sb.note}")

    print("\n[Known dimension cross-check - AUTO]")
    kd = detect_dimension_ratios_auto(bgr, dpi=dpi, given_ratio=given_scale_ratio, tol_pct=tol)
    print(f"  Found: {'YES' if kd.found else 'NO'}")
    if kd.implied_ratio is not None:
        print(f"  Implied ratio (median): {kd.implied_ratio:.2f}")
        print(f"  Compare with given: {given_scale_ratio:.2f}")
        print(f"  Result: {'PASS' if kd.pass_ok else 'FAIL'} (tol {tol:.1f}%)")
    print(f"  Note: {kd.note}")

    ok, reason = final_decision(sb, kd)
    print("\n[FINAL DECISION]")
    print(f"  Scale is {'VALID' if ok else 'INVALID'}")
    print(f"  Reason: {reason}")


if __name__ == "__main__":
    main()