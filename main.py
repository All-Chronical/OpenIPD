import math
import cv2
import CamFilters
import CR80Detect
import DebugWindow
import PupilDetect


def main():
    DebugWindow.setup_windows()
    camera = cv2.VideoCapture(0)
    cached_frame = None

    while True:
        ctrls = DebugWindow.get_controls()

        if not ctrls["freeze"] or cached_frame is None:
            ret, frame_read = camera.read()
            if not ret:
                break
            cached_frame = frame_read.copy()

        frame = cached_frame.copy()

        # Camera filtering
        frame_edges = CamFilters.apply_filters(
            frame,
            blur_val=ctrls["blur"],
            canny_low=ctrls["canny_low"],
            canny_high=ctrls["canny_high"],
        )
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Pupil and forehead ROI detection + 3D depth offset
        left_pupil, right_pupil, face_roi, depth_offset = PupilDetect.detect_pupils(frame)

        pupil_dist = None
        mid_pupil_x = None
        if left_pupil is not None and right_pupil is not None:
            pupil_dist = math.hypot(right_pupil[0] - left_pupil[0], right_pupil[1] - left_pupil[1])
            mid_pupil_x = (left_pupil[0] + right_pupil[0]) / 2.0

        # Mask edges outside forehead to eliminate background & face clutter
        if face_roi is not None:
            frame_edges = CamFilters.apply_roi_mask(frame_edges, face_roi)

        # CR80 card detection (constrained to forehead with scale prior and robust edge density)
        lines, quads, best_quad = CR80Detect.detect_cr80(
            frame_edges,
            theta_div=ctrls["theta_div"],
            hough_thresh=ctrls["hough_thresh"],
            min_length=ctrls["min_length"],
            max_gap=ctrls["max_gap"],
            angle_tol=ctrls["angle_tol"],
            corner_slack=ctrls["corner_slack"],
            overlap_pct=ctrls["overlap_pct"],
            infer_4th=ctrls["infer_4th"],
            pupil_dist=pupil_dist,
            frame_gray=frame_gray,
            depth_offset=depth_offset,
            mid_pupil_x=mid_pupil_x,
            roi=face_roi,
        )

        # Debug visualization
        debug_frame = DebugWindow.draw_debug_overlay(
            frame_edges,
            lines,
            quads,
            best_quad,
            left_pupil=left_pupil,
            right_pupil=right_pupil,
            face_roi=face_roi,
            depth_offset=depth_offset,
            freeze=ctrls["freeze"],
        )
        DebugWindow.show(debug_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key in (ord('f'), ord('F'), 32):
            DebugWindow.toggle_freeze()

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
