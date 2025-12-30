Install

pip install pymupdf

Run:

Correct Case:

$ python scale_check_auto_pymupdf.py dr.pdf
[OK] given_scale_ratio = 100.00  (example: 100.00 means 1:100)

[Scale bar check]
  Found: NO
  Note: Scale bar labels not detected in PDF text (common).

[Known dimension cross-check - AUTO estimate]
  Samples used: 110
  Estimated implied ratio (median): 100.00
  Compare with given: 100.00  (diff 0.00%)
  Result: PASS (tol 2.0%)

[FINAL DECISION - OR rule]
  Scale is ACCEPTED (at least one method passed).
  Confidence: MEDIUM (only one method passed).

Fail Case:


$ python scale_check_auto_pymupdf.py drawing.pdf
[OK] given_scale_ratio = 100.00  (example: 100.00 means 1:100)

[Scale bar check]
  Found: NO
  Note: Scale bar labels not detected in PDF text (common).

[Known dimension cross-check - AUTO estimate]
  Samples used: 117
  Estimated implied ratio (median): 104.72
  Compare with given: 100.00  (diff 4.72%)
  Result: FAIL (tol 2.0%)

[FINAL DECISION - OR rule]
  Scale is FLAGGED (both methods failed / not available).


