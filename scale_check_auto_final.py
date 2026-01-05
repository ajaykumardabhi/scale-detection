# scale_check_auto_final.py
# Works for: normal CAD PDFs + DWG->PDF exports (vector/text PDFs)
# Strategy:
# 1) Detect given scale ratio (1:100 etc.) from PDF TEXT using robust word-token parsing
# 2) Validate using known-dimension cross-check from PDF text + vector line segments
# 3) Optional OCR fallback (EasyOCR + OpenCV) only if scale text is not extractable
#
# Install:
#   Vector-only:  pip install pymupdf
#   With OCR:     pip install pymupdf opencv-python easyocr numpy pillow
#
# Run:
#   python scale_check_auto_final.py drawing.pdf --rule or
#   python scale_check_auto_final.py drawing.pdf --rule and
#   python scale_check_auto_final.py drawing.pdf --rule or --tol 5
#
# Notes:
# - "scale bar" check in vector mode is limited (label detection only). Many drawings have no scale bar.
# - Known-dimension check is the main validator for most architectural plans.
# - DWG->PDF often splits "Scale : 1 : 100" into multiple tokens; we handle that.

import re
import math
import statistics
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict

import fitz  # PyMuPDF

# Optional OCR deps
HAS_CV2 = False
HAS_EASYOCR = False
try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except Exception:
    pass

try:
    import easyocr
    HAS_EASYOCR = True
except Exception:
    pass


PT_TO_MM = 25.4 / 72.0


@dataclass
class MethodResult:
    method: str
    available: bool
    passed: bool
    implied_ratio: Optional[float] = None
    note: str = ""


def pct_diff(a: float, b: float) -> float:
    if b == 0:
        return 999.0
    return abs(a - b) / b * 100.0


def ratio_close(implied: float, given: float, tol_pct: float) -> bool:
    return pct_diff(implied, given) <= tol_pct


# --------------------------
# GIVEN SCALE DETECTION
# --------------------------

def parse_scale_ratio_from_text(text: str) -> Optional[float]:
    """
    Quick regex on whole text (works for many PDFs).
    """
    m = re.search(r"Scale\s*[:\-]?\s*1\s*[:/]\s*(\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
    if m:
        return float(m.group(1))

    m = re.search(r"\b1\s*[:/]\s*(\d+(?:\.\d+)?)\b", text)
    if m:
        return float(m.group(1))

    # Handle 'SCALE 1=100'
    m = re.search(r"Scale\s*[:\-]?\s*1\s*=\s*(\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
    if m:
        return float(m.group(1))

    return None


def parse_scale_ratio_from_words(page: fitz.Page) -> Optional[float]:
    """
    Robust scale parsing using tokenized words from PDF.
    Handles cases where DWG->PDF splits text: 'Scale' ':' '1' ':' '100'
    Also handles: 'SCALE' '1:100', '1' ':' '100', '1/100', 'SCALE 1=100', '1:100@A1' style tokens.
    """
    words = page.get_text("words") or []
    tokens = [str(w[4]).strip() for w in words if str(w[4]).strip()]
    tokens_l = [t.lower() for t in tokens]

    def to_float(x: str) -> Optional[float]:
        try:
            return float(x)
        except Exception:
            return None

    # Pass 1: locate "scale" then parse in a window
    for i, t in enumerate(tokens_l):
        if t == "scale":
            window = tokens[i:i + 15]
            joined = " ".join(window)

            # 1:100 or 1/100
            m = re.search(r"1\s*[:/]\s*(\d+(?:\.\d+)?)", joined)
            if m:
                return float(m.group(1))

            # 1=100
            m = re.search(r"1\s*=\s*(\d+(?:\.\d+)?)", joined)
            if m:
                return float(m.group(1))

            # Token pattern Scale : 1 : 100 (or /)
            for j in range(i, min(i + 12, len(tokens) - 4)):
                if tokens[j].lower() != "scale":
                    continue
                # Accept optional ":" after scale
                k = j + 1
                if k < len(tokens) and tokens[k] in (":",):
                    k += 1
                if k + 2 >= len(tokens):
                    continue
                if tokens[k] in ("1", "1.0") and tokens[k + 1] in (":", "/"):
                    v = to_float(tokens[k + 2])
                    if v and v > 0:
                        return float(v)

    # Pass 2: direct token contains 1:100 or 1/100, sometimes appended with @A1
    for t in tokens:
        m = re.search(r"\b1\s*[:/]\s*(\d+(?:\.\d+)?)\b", t)
        if m:
            return float(m.group(1))
        m = re.search(r"\b1[:/](\d+(?:\.\d+)?)\b", t)
        if m:
            return float(m.group(1))

    # Pass 3: anywhere 1 : 100 sequence without 'scale'
    for i in range(len(tokens) - 2):
        if tokens[i] == "1" and tokens[i + 1] in (":", "/"):
            v = to_float(tokens[i + 2])
            if v and v > 0:
                return float(v)

    return None


def debug_find_scale_lines(page: fitz.Page, limit: int = 25) -> None:
    """
    Print lines that contain 'scale' or a 1:xxx-like pattern (debug helper).
    """
    txt = page.get_text("text") or ""
    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    hits = []
    for ln in lines:
        if re.search(r"scale", ln, re.IGNORECASE) or re.search(r"\b1\s*[:/]\s*\d+\b", ln):
            hits.append(ln)
    print("\n[DEBUG SCALE LINES]")
    if not hits:
        print("  (no obvious lines found)")
    else:
        for ln in hits[:limit]:
            print(" ", ln)


# --------------------------
# PDF DIAGNOSTICS
# --------------------------

def pdf_diagnostics(page: fitz.Page) -> Dict[str, int]:
    words = page.get_text("words") or []
    drawings = page.get_drawings() or []
    images = page.get_images(full=True) or []
    return {"words": len(words), "drawings": len(drawings), "images": len(images)}


# --------------------------
# VECTOR/TEXT MODE
# --------------------------

def extract_numeric_words(page: fitz.Page) -> List[Tuple[int, float, float]]:
    """
    Candidate dimension-like numbers from PDF text layer:
      returns [(value_mm, cx, cy), ...]
    """
    words = page.get_text("words") or []
    out = []
    for x0, y0, x1, y1, w, *_ in words:
        w2 = str(w).replace(",", "").strip()
        if re.fullmatch(r"\d{3,6}", w2):
            v = int(w2)
            # mm-like range
            if 1000 <= v <= 200000:
                cx = (x0 + x1) / 2.0
                cy = (y0 + y1) / 2.0
                out.append((v, cx, cy))
    out.sort(key=lambda t: t[0], reverse=True)
    return out


def extract_axis_aligned_segments(page: fitz.Page, min_len_pt: float = 60.0) -> List[Tuple[float, float, float, float, float]]:
    """
    Extract long horizontal/vertical line segments from vector drawing layer.
    Returns [(x0,y0,x1,y1,length_pt), ...]
    """
    segs = []
    drawings = page.get_drawings() or []
    for dr in drawings:
        width = dr.get("width", 1.0)
        # Skip thick lines; dims are usually thin
        if width is not None and width > 3.0:
            continue
        for it in dr.get("items", []):
            if not it:
                continue
            if it[0] != "l":
                continue
            p1, p2 = it[1], it[2]
            x0, y0, x1, y1 = p1.x, p1.y, p2.x, p2.y
            L = math.hypot(x1 - x0, y1 - y0)
            if L < min_len_pt:
                continue
            if abs(y1 - y0) < 1.0 or abs(x1 - x0) < 1.0:
                segs.append((x0, y0, x1, y1, L))
    return segs


def closest_seg_length(cx: float, cy: float, segs: List[Tuple[float, float, float, float, float]], max_dist_pt: float = 25.0) -> Optional[float]:
    """
    Find nearest axis-aligned segment to point (cx,cy).
    Returns segment length (pt) if close enough.
    """
    best = None
    for x0, y0, x1, y1, L in segs:
        # Horizontal
        if abs(y1 - y0) < 1.0:
            xmin, xmax = sorted([x0, x1])
            dx = 0.0
            if cx < xmin:
                dx = xmin - cx
            elif cx > xmax:
                dx = cx - xmax
            dist = math.hypot(dx, cy - y0)
        # Vertical
        elif abs(x1 - x0) < 1.0:
            ymin, ymax = sorted([y0, y1])
            dy = 0.0
            if cy < ymin:
                dy = ymin - cy
            elif cy > ymax:
                dy = cy - ymax
            dist = math.hypot(cx - x0, dy)
        else:
            continue

        if dist <= max_dist_pt and (best is None or dist < best[0]):
            best = (dist, L)

    return best[1] if best else None


def vector_known_dimension_check(page: fitz.Page, given_ratio: float, tol_pct: float) -> MethodResult:
    """
    Improved known-dimension check for DWG->PDF:
    - For each dimension text, compute expected paper length from given_ratio
    - Search multiple nearby vector segments
    - Choose the segment whose length best matches the expected paper length
    - Take median implied ratio from best matches
    """

    nums = extract_numeric_words(page)
    segs = extract_axis_aligned_segments(page)

    if not nums:
        return MethodResult("known_dimension_vector", available=False, passed=False, note="No numeric dimensions in PDF text layer.")
    if not segs:
        return MethodResult("known_dimension_vector", available=False, passed=False, note="No usable vector line segments (get_drawings empty/filtered).")

    # Precompute segment midpoints for faster distance checks
    seg_meta = []
    for x0, y0, x1, y1, L in segs:
        mx = (x0 + x1) / 2.0
        my = (y0 + y1) / 2.0
        seg_meta.append((x0, y0, x1, y1, L, mx, my))

    implied_list = []
    used_pairs = 0

    # Use top larger dimensions first (outer dims are more reliable)
    for val_mm, cx, cy in nums[:300]:
        # Expected paper length at given scale
        expected_paper_mm = val_mm / given_ratio
        expected_paper_pt = expected_paper_mm / PT_TO_MM

        # Find candidate segments near this text
        candidates = []
        for x0, y0, x1, y1, L, mx, my in seg_meta:
            d = math.hypot(mx - cx, my - cy)
            if d > 80:  # search radius in points (increase if needed)
                continue

            # score: closeness to expected length + closeness to label
            length_err = abs(L - expected_paper_pt) / max(expected_paper_pt, 1e-6)
            score = (length_err * 2.0) + (d / 80.0)  # weight length more than distance
            candidates.append((score, L, d))

        if not candidates:
            continue

        # Pick best candidate (lowest score)
        candidates.sort(key=lambda t: t[0])
        best_score, best_L, best_d = candidates[0]

        # Reject obviously wrong matches (length too far from expected)
        # This is the key improvement for DWG->PDF noisy geometry.
        if abs(best_L - expected_paper_pt) / max(expected_paper_pt, 1e-6) > 0.25:
            # more than 25% off expected paper length → likely wrong line
            continue

        paper_mm = best_L * PT_TO_MM
        implied = val_mm / paper_mm if paper_mm > 0 else None
        if implied is None:
            continue

        if 20.0 <= implied <= 300.0:
            implied_list.append(implied)
            used_pairs += 1

    if len(implied_list) < 8:
        return MethodResult(
            "known_dimension_vector",
            available=False,
            passed=False,
            note=f"Not enough reliable dimension↔line matches after filtering (kept {len(implied_list)}). Try increasing search radius or exporting DWG with true dimension entities."
        )

    # Robust central value
    implied_med = statistics.median(implied_list)

    ok = ratio_close(implied_med, given_ratio, tol_pct)
    note = f"median={implied_med:.2f}, kept={len(implied_list)}, scanned_dims={min(300, len(nums))}"
    return MethodResult("known_dimension_vector", available=True, passed=ok, implied_ratio=implied_med, note=note)

def vector_scale_bar_check(page: fitz.Page, given_ratio: float, tol_pct: float) -> MethodResult:
    """
    Only detects presence of typical scale bar label patterns in text layer.
    Full bar measurement varies a lot; prefer known-dimension check for validation.
    """
    text = page.get_text("text") or ""
    patterns = [
        r"\b0\s+1\s+2\s+3\s+4\s+5\s*(m|meter|metre)\b",
        r"\b0\s+2\s+4\s+6\s+8\s+10\s*(m|meter|metre)\b",
        r"\b0\s+5\s+10\s*(m|meter|metre)\b",
        r"\b0\s+10\s+20\s*(m|meter|metre)\b",
    ]
    for p in patterns:
        if re.search(p, text, flags=re.IGNORECASE):
            return MethodResult("scale_bar_vector", available=True, passed=False,
                                note="Scale bar label pattern found in text, but measuring bar geometry is not implemented in vector-only mode.")
    return MethodResult("scale_bar_vector", available=False, passed=False, note="Scale bar label not detected in PDF text layer (common).")


# --------------------------
# OCR FALLBACK MODE
# --------------------------

def render_page_to_bgr(doc: fitz.Document, page_index: int, dpi: int = 300):
    page = doc[page_index]
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)  # RGB
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return bgr


def ocr_extract_given_scale_ratio(doc: fitz.Document, page_index: int, dpi: int = 300) -> Optional[float]:
    if not (HAS_CV2 and HAS_EASYOCR):
        return None
    bgr = render_page_to_bgr(doc, page_index, dpi=dpi)
    reader = easyocr.Reader(["en"], gpu=False)

    h, w = bgr.shape[:2]
    # Title block often bottom-right
    roi = bgr[int(h * 0.65):h, int(w * 0.55):w].copy()
    rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    words = reader.readtext(rgb, detail=0)
    text = " ".join(words).replace("I", "1")

    m = re.search(r"Scale\s*[:\-]?\s*1\s*[:/]\s*(\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
    if m:
        return float(m.group(1))
    m = re.search(r"\b1\s*[:/]\s*(\d+(?:\.\d+)?)\b", text)
    if m:
        return float(m.group(1))
    m = re.search(r"Scale\s*[:\-]?\s*1\s*=\s*(\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
    if m:
        return float(m.group(1))
    return None


# --------------------------
# FINAL DECISION
# --------------------------

def decide_final(scale_bar: MethodResult, dim_check: MethodResult, use_or: bool) -> Tuple[bool, str]:
    avail = [r for r in (scale_bar, dim_check) if r.available]
    if not avail:
        return False, "No validation method available."

    if use_or:
        if any(r.passed for r in avail):
            if len(avail) == 2 and all(r.passed for r in avail):
                return True, "VALID (both checks PASS)."
            return True, "VALID (OR rule: at least one available check PASS)."
        return False, "INVALID (OR rule: all available checks FAIL)."

    # AND rule: if both are available, both must pass
    if scale_bar.available and dim_check.available:
        if scale_bar.passed and dim_check.passed:
            return True, "VALID (both checks PASS)."
        return False, "INVALID (mismatch/fail between scale bar and known-dimension)."

    # Fallback to the one available
    only = dim_check if dim_check.available else scale_bar
    if only.passed:
        return True, f"VALID (only {only.method} available and PASS)."
    return False, f"INVALID (only {only.method} available and FAIL)."


def print_result(given_ratio: float, r: MethodResult, tol_pct: float) -> None:
    print(f"\n[{r.method}]")
    print(f"  Available: {'YES' if r.available else 'NO'}")
    if r.implied_ratio is not None:
        print(f"  Implied ratio: {r.implied_ratio:.2f}")
        print(f"  Compare with given: {given_ratio:.2f}")
        print(f"  Diff: {pct_diff(r.implied_ratio, given_ratio):.2f}% (tol {tol_pct:.1f}%)")
    print(f"  Result: {'PASS' if r.passed else 'FAIL'}")
    if r.note:
        print(f"  Note: {r.note}")


def main(pdf_path: str, page_index: int = 0, tol_pct: float = 2.0, dpi: int = 300, rule: str = "or") -> int:
    use_or = (rule.lower() == "or")

    doc = fitz.open(pdf_path)
    page = doc[page_index]

    diag = pdf_diagnostics(page)
    print("[PDF DIAGNOSTICS]")
    print(f"  WORDS: {diag['words']}")
    print(f"  DRAWINGS: {diag['drawings']}")
    print(f"  IMAGES: {diag['images']}")

    # Given scale ratio detection (vector/text first)
    text = page.get_text("text") or ""
    given_ratio = parse_scale_ratio_from_text(text)
    if given_ratio is None:
        given_ratio = parse_scale_ratio_from_words(page)

    # OCR fallback only if still not found
    if given_ratio is None:
        given_ratio = ocr_extract_given_scale_ratio(doc, page_index, dpi=dpi)

    if given_ratio is None:
        debug_find_scale_lines(page)
        print("\n[FAIL] Could not detect given scale ratio (1:XX) from text or OCR.")
        print("      If this is DWG->PDF, ensure scale is written in title block OR pass the scale manually in your pipeline.")
        doc.close()
        return 2

    print(f"\n[OK] given_scale_ratio = {given_ratio:.2f}  (example: 100.00 means 1:100)")
    print(f"[Settings] tol={tol_pct:.1f}% | dpi={dpi} | final_rule={'OR' if use_or else 'AND'}")

    # Choose mode: vector mode is preferred when we have plenty of text+drawings
    use_vector_mode = (diag["words"] >= 20 and diag["drawings"] >= 20)
    print(f"\n[MODE] {'VECTOR/TEXT (PyMuPDF)' if use_vector_mode else 'OCR (fallback)'}")

    if use_vector_mode:
        sb = vector_scale_bar_check(page, given_ratio, tol_pct)
        kd = vector_known_dimension_check(page, given_ratio, tol_pct)
    else:
        # We only do OCR scale extraction in this file; dimension OCR is not included here
        # because your DWG->PDF is vector already (based on your diagnostics).
        sb = MethodResult("scale_bar_ocr", available=False, passed=False, note="OCR scale bar check not enabled in this vector-first build.")
        kd = MethodResult("known_dimension_ocr", available=False, passed=False, note="OCR known-dimension check not enabled in this vector-first build.")

    print_result(given_ratio, sb, tol_pct)
    print_result(given_ratio, kd, tol_pct)

    ok, reason = decide_final(sb, kd, use_or=use_or)
    print("\n[FINAL DECISION]")
    print(f"  Scale is {'VALID' if ok else 'INVALID'}")
    print(f"  Reason: {reason}")

    doc.close()
    return 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", help="Path to drawing PDF")
    ap.add_argument("--page", type=int, default=0, help="Page index (default 0)")
    ap.add_argument("--tol", type=float, default=2.0, help="Tolerance percent (default 2.0)")
    ap.add_argument("--dpi", type=int, default=300, help="DPI (only used for OCR fallback)")
    ap.add_argument("--rule", choices=["or", "and"], default="or", help="Final rule")
    args = ap.parse_args()

    raise SystemExit(main(args.pdf, page_index=args.page, tol_pct=args.tol, dpi=args.dpi, rule=args.rule))
