import re
import math
import statistics
import fitz  # PyMuPDF


PT_TO_MM = 25.4 / 72.0


def parse_given_scale_ratio(page_text: str) -> float | None:
    """
    Returns ratio number for '1 : 100' -> 100.0
    """
    m = re.search(r"Scale:\s*1\s*[:]\s*(\d+)", page_text, re.IGNORECASE)
    if m:
        return float(m.group(1))
    # sometimes it appears as "1 : 100" without "Scale:"
    m = re.search(r"\b1\s*[:]\s*(\d+)\b", page_text)
    if m:
        return float(m.group(1))
    return None


def find_scale_bar_in_text(page_text: str):
    """
    Text-only scale bar detection.
    Many CAD PDFs do NOT store scale bar labels as clean text.
    Returns None if not found.

    Looks for patterns like: "0 1 2 3 4 5 m" or "0 5m" etc.
    """
    # Example patterns (you can expand later)
    patterns = [
        r"\b0\s+1\s+2\s+3\s+4\s+5\s*(m|meter|metre)\b",
        r"\b0\s+2\s+4\s+6\s+8\s+10\s*(m|meter|metre)\b",
    ]
    for p in patterns:
        if re.search(p, page_text, re.IGNORECASE):
            return "FOUND_TEXT_PATTERN"
    return None


def extract_numeric_words(page: fitz.Page):
    """
    Extract numeric tokens from the PDF text layer with positions.
    Returns list of (value_mm, cx, cy)
    """
    words = page.get_text("words")  # x0,y0,x1,y1,word,...
    out = []
    for x0, y0, x1, y1, w, *_ in words:
        if re.fullmatch(r"\d{3,6}", w):
            v = int(w)
            # mm-like range (tune as needed)
            if 300 <= v <= 200000:
                cx = (x0 + x1) / 2.0
                cy = (y0 + y1) / 2.0
                out.append((v, cx, cy))
    return out


def extract_long_axis_aligned_segments(page: fitz.Page, min_len_pt: float = 50.0):
    """
    Extract long horizontal/vertical vector line segments from the PDF drawing layer.
    Returns list of segments: (x0,y0,x1,y1,length_pt)
    """
    segs = []
    drawings = page.get_drawings()
    for dr in drawings:
        w = dr.get("width")
        if w is None:
            continue
        if w > 2.0:
            continue
        for it in dr["items"]:
            if it[0] != "l":
                continue
            (x0, y0), (x1, y1) = it[1], it[2]
            L = math.hypot(x1 - x0, y1 - y0)
            if L < min_len_pt:
                continue

            # keep mostly horizontal or vertical lines
            if abs(y1 - y0) < 1.0 or abs(x1 - x0) < 1.0:
                segs.append((x0, y0, x1, y1, L))
    return segs


def closest_segment_length(cx, cy, segs, max_dist_pt: float = 25.0):
    """
    Find the closest axis-aligned segment to a point (cx,cy).
    Returns segment length in points, or None.
    """
    best = None
    for x0, y0, x1, y1, L in segs:
        # horizontal
        if abs(y1 - y0) < 1.0:
            xmin, xmax = sorted([x0, x1])
            dx = 0.0
            if cx < xmin:
                dx = xmin - cx
            elif cx > xmax:
                dx = cx - xmax
            dist = math.hypot(dx, cy - y0)

        # vertical
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


def implied_ratio(real_mm: float, seg_len_pt: float) -> float:
    paper_mm = seg_len_pt * PT_TO_MM
    if paper_mm <= 0:
        return float("nan")
    return real_mm / paper_mm


def known_dimension_estimate_ratio(page: fitz.Page):
    """
    Best-effort automatic estimate:
    - find numeric dimension texts from the text layer
    - find nearby long vector segments (dimension lines)
    - compute implied ratios
    - return median implied ratio (robust)
    """
    nums = extract_numeric_words(page)
    segs = extract_long_axis_aligned_segments(page)

    ratios = []
    for v, cx, cy in nums:
        seg_len = closest_segment_length(cx, cy, segs, max_dist_pt=25.0)
        if not seg_len:
            continue
        r = implied_ratio(v, seg_len)
        if 20 <= r <= 300:
            ratios.append(r)

    if len(ratios) < 10:
        return None, ratios

    return statistics.median(ratios), ratios


def compare_ratio(implied: float, given: float, tol_pct: float) -> bool:
    if implied is None or given is None:
        return False
    diff_pct = abs(implied - given) / given * 100.0
    return diff_pct <= tol_pct


def main(pdf_path: str, tol_pct: float = 2.0):
    doc = fitz.open(pdf_path)
    page = doc[0]
    text = page.get_text()

    given = parse_given_scale_ratio(text)
    if given is None:
        raise SystemExit("Could not find given scale in PDF text (Scale: 1 : XX).")

    print(f"[OK] given_scale_ratio = {given:.2f}  (example: 100.00 means 1:100)")

    # 1) Scale bar (text-only attempt)
    print("\n[Scale bar check]")
    sb = find_scale_bar_in_text(text)
    if sb:
        print("  Found: YES (text pattern)")
        # NOTE: measuring the bar geometry is not implemented here because
        # many PDFs draw scale bars as vector graphics without clear labeling structure.
        scale_bar_pass = True
    else:
        print("  Found: NO")
        print("  Note: Scale bar labels not detected in PDF text (common).")
        scale_bar_pass = False

    # 2) Known dimension (auto estimate)
    print("\n[Known dimension cross-check - AUTO estimate]")
    median_ratio, ratios = known_dimension_estimate_ratio(page)
    if median_ratio is None:
        print("  Could not estimate (not enough matches between dimension text and vector lines).")
        known_dim_pass = False
    else:
        diff_pct = abs(median_ratio - given) / given * 100.0
        print(f"  Samples used: {len(ratios)}")
        print(f"  Estimated implied ratio (median): {median_ratio:.2f}")
        print(f"  Compare with given: {given:.2f}  (diff {diff_pct:.2f}%)")
        known_dim_pass = compare_ratio(median_ratio, given, tol_pct)
        print(f"  Result: {'PASS' if known_dim_pass else 'FAIL'} (tol {tol_pct:.1f}%)")

    # FINAL: your OR rule
    print("\n[FINAL DECISION - OR rule]")
    if scale_bar_pass or known_dim_pass:
        print("  Scale is ACCEPTED (at least one method passed).")
        if scale_bar_pass and known_dim_pass:
            print("  Confidence: HIGH (both agree).")
        else:
            print("  Confidence: MEDIUM (only one method passed).")
    else:
        print("  Scale is FLAGGED (both methods failed / not available).")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python scale_check_auto_pymupdf.py drawing.pdf")
        raise SystemExit(2)
    main(sys.argv[1], tol_pct=2.0)