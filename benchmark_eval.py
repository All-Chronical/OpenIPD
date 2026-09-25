import glob
import math
import os
import cv2
import numpy as np

import CamFilters
import CR80Detect
import DebugWindow
import PupilDetect

bench_dir = r"d:\Projects\current\benchmark test"
images = sorted(glob.glob(os.path.join(bench_dir, "*.jpg")))
print(f"Found {len(images)} benchmark images\n")

GROUND_TRUTH_IPD = 60.5

results = []

for img_path in images:
    name = os.path.basename(img_path)
    frame = cv2.imread(img_path)
    if frame is None:
        print(f"{name}: Failed to read")
        continue

    h, w = frame.shape[:2]

    # Reset state for each test photo
    PupilDetect._pupil_tracker = PupilDetect.PupilTracker()
    CR80Detect._card_tracker = CR80Detect.ParametricCardTracker()

    left_p, right_p, roi, depth_offset = PupilDetect.detect_pupils(frame)
    pupil_dist = None
    mid_pupil_x = None
    if left_p and right_p:
        pupil_dist = math.hypot(right_p[0] - left_p[0], right_p[1] - left_p[1])
        mid_pupil_x = (left_p[0] + right_p[0]) / 2.0

    # Current default controls
    edges = CamFilters.apply_filters(frame, blur_val=7, canny_low=35, canny_high=70)
    masked_edges = CamFilters.apply_roi_mask(edges, roi)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    lines, quads, best_quad = CR80Detect.detect_cr80(
        masked_edges,
        theta_div=549,
        hough_thresh=42,
        min_length=20,
        max_gap=25,
        angle_tol=7,
        corner_slack=150,
        overlap_pct=0.0,
        infer_4th=True,
        pupil_dist=pupil_dist,
        frame_gray=frame_gray,
        depth_offset=depth_offset,
        mid_pupil_x=mid_pupil_x,
        roi=roi,
    )

    card_len = CR80Detect.get_perspective_flattened_long_side(best_quad)
    raw_ipd = None
    cal_ipd = None
    if pupil_dist and card_len:
        raw_ipd = (pupil_dist / card_len) * 85.60
        cal_ipd = CR80Detect.compute_calibrated_ipd(pupil_dist, card_len, depth_offset=depth_offset)

    err_str = f"diff={cal_ipd - GROUND_TRUTH_IPD:+.2f}mm" if cal_ipd is not None else "N/A"
    p_str = f"{pupil_dist:.1f}px" if pupil_dist is not None else "None"
    c_str = f"{card_len:.1f}px" if card_len is not None else "None"
    r_str = f"{raw_ipd:.2f}mm" if raw_ipd is not None else "None"
    cal_str = f"{cal_ipd:.2f}mm" if cal_ipd is not None else "None"

    print(f"{name:<30} | Pupils: {p_str:<8} | Card: {c_str:<8} | Raw: {r_str:<7} | IPD: {cal_str:<7} | {err_str}")
    results.append({
        "name": name,
        "pupil_dist": pupil_dist,
        "card_len": card_len,
        "raw_ipd": raw_ipd,
        "cal_ipd": cal_ipd,
        "best_quad": best_quad,
        "roi": roi,
    })

valid_ipds = [r["cal_ipd"] for r in results if r["cal_ipd"] is not None]
raw_ipds = [r["raw_ipd"] for r in results if r["raw_ipd"] is not None]
if valid_ipds:
    print("\n" + "=" * 80)
    print(f"Cards detected in {len(valid_ipds)} / {len(images)} benchmark images")
    print(f"Mean Calibrated IPD: {np.mean(valid_ipds):.2f} mm (std: {np.std(valid_ipds):.2f} mm, min: {np.min(valid_ipds):.2f}, max: {np.max(valid_ipds):.2f})")
    print(f"Mean Raw IPD       : {np.mean(raw_ipds):.2f} mm (std: {np.std(raw_ipds):.2f} mm)")
    print(f"Medically Measured : {GROUND_TRUTH_IPD:.2f} mm")
    print(f"Error of Mean IPD  : {np.mean(valid_ipds) - GROUND_TRUTH_IPD:+.2f} mm")
    print("=" * 80)
else:
    print("\nNo cards detected in any image!")
