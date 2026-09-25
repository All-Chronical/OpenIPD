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

        # Pupil and face/forehead ROI detection
        left_pupil, right_pupil, face_roi = PupilDetect.detect_pupils(frame)

        # Mask edges outside face/forehead to eliminate background clutter
        if face_roi is not None:
            frame_edges = CamFilters.apply_roi_mask(frame_edges, face_roi)

        # CR80 card detection (constrained to ROI)
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
