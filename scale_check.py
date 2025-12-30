import re
import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

import fitz  # PyMuPDF
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Helpers
# -----------------------------
PT_TO_MM = 25.4 / 72.0  # PDF points to mm

def parse_scale_to_ratio(scale_text: str) -> Optional[float]:
    """
    Extracts ratio from strings like:
      "Scale: 1 : 100", "1:100", "1 : 50"
    Returns numeric ratio (e.g., 100.0) or None.
    """
    if not scale_text:
        return None
    m = re.search(r"\b1\s*[:]\s*(\d+(?:\.\d+)?)\b", scale_text)
    if not m:
        m = re.search(r"\b1\s*[:]\s*(\d+)\b", scale_text)
    if m:
        return float(m.group(1))
    # Sometimes written "1 : 100"
    m = re.search(r"\b1\s*[:]\s*(\d+(?:\.\d+)?)\b", scale_text.replace(" ", ""))
    if m:
        return float(m.group(1))
    return None

def extract_given_scale_ratio_from_pdf(pdf_path: str, page_no: int = 0) -> Optional[float]:
    """
    Pulls text from PDF and tries to find 'Scale: 1 : 100' etc.
    """
    doc = fitz.open(pdf_path)
    page = doc[page_no]
    text = page.get_text("text") or ""
    # Look for the most relevant "Scale" line
    # Example in your drawing: "Scale: 1 : 100" and also "1 : 100" and "Scale 1 : 100"
    candidates = []
    for line in text.splitlines():
        if "scale" in line.lower() or re.search(r"\b1\s*[:]\s*\d+", line):
            candidates.append(line.strip())

    joined = " | ".join(candidates) if candidates else text
    ratio = parse_scale_to_ratio(joined)
    doc.close()
    return ratio

def render_page_to_image(pdf_path: str, page_no: int = 0, dpi: int = 200) -> np.ndarray:
    """
    Renders a PDF page to an RGB numpy array.
    """
    doc = fitz.open(pdf_path)
    page = doc[page_no]
    mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    doc.close()
    return img

def click_two_points(img: np.ndarray, title: str) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Lets user click two points on the rendered page image.
    """
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_title(title)
    pts = plt.ginput(2, timeout=0)
    plt.close(fig)

    if len(pts) != 2:
        raise RuntimeError("You must click exactly 2 points.")
    (x1, y1), (x2, y2) = pts
    return (x1, y1), (x2, y2)

def px_distance(p1, p2) -> float:
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def px_to_mm(px: float, dpi: float) -> float:
    # 1 inch = 25.4 mm; px/dpi = inches
    return (px / dpi) * 25.4

def ratio_close(a: float, b: float, tol_pct: float = 2.0) -> bool:
    """
    True if a and b differ by <= tol_pct percent.
    """
    if a == 0 or b == 0:
        return False
    return abs(a - b) / b * 100.0 <= tol_pct


# -----------------------------
# Scale bar detection (best-effort)
# -----------------------------
@dataclass
class ScaleBarResult:
    found: bool
    implied_ratio: Optional[float] = None
    real_length_mm: Optional[float] = None
    paper_length_mm: Optional[float] = None
    note: str = ""

def try_detect_scale_bar_from_pdf_text(pdf_path: str, page_no: int = 0) -> ScaleBarResult:
    """
    Best-effort detection:
    Many PDFs include scale bar labels like: '0 2 4 6 8 10 m' near a bar.
    We try to find a cluster containing 'm' and several numbers in the same line area.
    If found, estimate bar span using min/max x of the numeric labels.
    This won't work on all drawings.
    """
    doc = fitz.open(pdf_path)
    page = doc[page_no]
    words = page.get_text("words")  # list of (x0, y0, x1, y1, word, block, line, word_no)
    doc.close()

    if not words:
        return ScaleBarResult(found=False, note="No text words extracted from PDF.")

    # Group by (block, line)
    from collections import defaultdict
    lines = defaultdict(list)
    for w in words:
        x0, y0, x1, y1, txt, block, line, wno = w
        lines[(block, line)].append(w)

    # Look for a line containing "m" and at least 3 numeric labels (0,2,5,10 etc.)
    best = None
    for (block, line), ws in lines.items():
        texts = [w[4] for w in ws]
        has_m = any(t.lower() == "m" or t.lower().endswith("m") for t in texts)
        nums = []
        for t in texts:
            t2 = t.lower().replace("m", "")
            if re.fullmatch(r"\d+(\.\d+)?", t2):
                nums.append(float(t2))
        if has_m and len(nums) >= 3 and min(nums) == 0:
            # Candidate scale bar label line
            # Use min/max x of numeric words as a proxy for bar length
            numeric_words = []
            for w in ws:
                t = w[4].lower()
                t2 = t.replace("m", "")
                if re.fullmatch(r"\d+(\.\d+)?", t2):
                    numeric_words.append(w)
            xs = [w[0] for w in numeric_words] + [w[2] for w in numeric_words]
            x_min, x_max = min(xs), max(xs)
            # Real length is max numeric value in meters (assume meters if 'm' present)
            real_len_m = max(nums)
            best = (real_len_m, x_min, x_max)
            break

    if not best:
        return ScaleBarResult(found=False, note="Scale bar labels not detected in PDF text (common).")

    real_len_m, x_min, x_max = best
    paper_len_pt = (x_max - x_min)
    paper_len_mm = paper_len_pt * PT_TO_MM
    real_len_mm = real_len_m * 1000.0

    if paper_len_mm <= 0:
        return ScaleBarResult(found=False, note="Detected scale-bar region had non-positive length.")

    implied_ratio = real_len_mm / paper_len_mm
    return ScaleBarResult(
        found=True,
        implied_ratio=implied_ratio,
        real_length_mm=real_len_mm,
        paper_length_mm=paper_len_mm,
        note=f"Detected labels up to {real_len_m} m; estimated span from label positions."
    )


# -----------------------------
# Main validation pipeline
# -----------------------------
def validate_scale_pipeline(
    pdf_path: str,
    page_no: int = 0,
    dpi: int = 220,
    tol_pct: float = 2.0
):
    # 1) Extract given scale from PDF -> store in variable
    given_scale_ratio = extract_given_scale_ratio_from_pdf(pdf_path, page_no=page_no)
    if not given_scale_ratio:
        raise RuntimeError("Could not auto-detect given scale ratio from PDF text.")

    print(f"[OK] given_scale_ratio = {given_scale_ratio:.2f}  (example: 100.00 means 1:100)")

    # 2) First validate using scale bar (best effort)
    sb = try_detect_scale_bar_from_pdf_text(pdf_path, page_no=page_no)
    scale_bar_pass = False
    if sb.found and sb.implied_ratio:
        scale_bar_pass = ratio_close(sb.implied_ratio, given_scale_ratio, tol_pct=tol_pct)
        print("\n[Scale bar check]")
        print(f"  Found: YES")
        print(f"  Implied ratio: {sb.implied_ratio:.2f}")
        print(f"  Compare with given: {given_scale_ratio:.2f}")
        print(f"  Result: {'PASS' if scale_bar_pass else 'FAIL'} (tol {tol_pct:.1f}%)")
        print(f"  Note: {sb.note}")
    else:
        print("\n[Scale bar check]")
        print("  Found: NO")
        print(f"  Note: {sb.note}")

    # 3) Cross-check using known dimensions (interactive click)
    img = render_page_to_image(pdf_path, page_no=page_no, dpi=dpi)

    print("\n[Known dimension cross-check]")
    print("  You will click TWO endpoints of a dimension you trust (e.g., a long outer dimension line).")
    print("  Then you will enter the dimension value in mm (example: 37522).")

    p1, p2 = click_two_points(img, "Click 2 endpoints of a known dimension (then close window)")
    measured_px = px_distance(p1, p2)
    measured_mm_on_paper = px_to_mm(measured_px, dpi=dpi)

    dim_mm = float(input("Enter the known dimension value (in mm, as written on drawing): ").strip())
    implied_ratio_dim = dim_mm / measured_mm_on_paper

    dim_pass = ratio_close(implied_ratio_dim, given_scale_ratio, tol_pct=tol_pct)

    print(f"  Measured paper length (mm): {measured_mm_on_paper:.2f}")
    print(f"  Real length from dimension (mm): {dim_mm:.2f}")
    print(f"  Implied ratio: {implied_ratio_dim:.2f}")
    print(f"  Compare with given: {given_scale_ratio:.2f}")
    print(f"  Result: {'PASS' if dim_pass else 'FAIL'} (tol {tol_pct:.1f}%)")

    # 4) Final decision using your rule:
    # ✔ First validate using scale bar
    # ✔ Cross-check using known dimensions
    # ✔ If both agree → scale is correct
    # ✔ If mismatch → flag drawing as invalid
    #
    # Practical handling:
    # - If scale bar not found, final decision uses dimension only.
    if sb.found:
        final_ok = (scale_bar_pass and dim_pass)
        reason = "Both scale bar and known-dimension checks agree." if final_ok else "Mismatch between scale bar and/or known-dimension check."
    else:
        final_ok = dim_pass
        reason = "Scale bar not found; decision based on known-dimension check only."

    print("\n[FINAL DECISION]")
    print(f"  Scale is {'CORRECT' if final_ok else 'INVALID'}")
    print(f"  Reason: {reason}")

    return {
        "given_scale_ratio": given_scale_ratio,
        "scale_bar_found": sb.found,
        "scale_bar_implied_ratio": sb.implied_ratio,
        "scale_bar_pass": scale_bar_pass if sb.found else None,
        "dim_implied_ratio": implied_ratio_dim,
        "dim_pass": dim_pass,
        "final_ok": final_ok,
        "note": sb.note
    }


if __name__ == "__main__":
    PDF_PATH = r"drawing.pdf"
    validate_scale_pipeline(PDF_PATH, page_no=0, dpi=220, tol_pct=2.0)
