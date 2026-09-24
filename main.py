import cv2
import math
import numpy as np

REFERENCE_CARD = np.array([
    [-42.80, -26.99],
    [ 42.80, -26.99],
    [ 42.80,  26.99],
    [-42.80,  26.99]
], dtype=np.float32)


def line_angle(line):
    x1, y1, x2, y2 = line
    return math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180


def is_perpendicular(ang1, ang2, tol=15):
    diff = abs(((ang1 - ang2 + 90) % 180) - 90)
    return abs(diff - 90) <= tol


def line_intersect(s1, s2):
    p, r = s1[:2], s1[2:] - s1[:2]
    q, s = s2[:2], s2[2:] - s2[:2]
    den = r[0] * s[1] - r[1] * s[0]
    if abs(den) < 1e-6:
        return None
    t = ((q - p)[0] * s[1] - (q - p)[1] * s[0]) / den
    return p + t * r


def find_candidate_quads(lines):
    if lines is None or len(lines) < 4:
        return []

    segs = [l.flatten().astype(float) for l in lines]
    angles = [line_angle(s) for s in segs]

    quads = []
    n = len(segs)
    for i in range(n):
        for j in range(i + 1, n):
            if not is_perpendicular(angles[i], angles[j]):
                continue
            for k in range(i + 1, n):
                if not is_perpendicular(angles[j], angles[k]):
                    continue
                for m in range(j + 1, n):
                    if not is_perpendicular(angles[k], angles[m]) or not is_perpendicular(angles[m], angles[i]):
                        continue
                    c0 = line_intersect(segs[i], segs[j])
                    c1 = line_intersect(segs[j], segs[k])
                    c2 = line_intersect(segs[k], segs[m])
                    c3 = line_intersect(segs[m], segs[i])
                    if all(c is not None for c in (c0, c1, c2, c3)):
                        quads.append(np.array([c0, c1, c2, c3], dtype=np.int32))
    return quads


def normalize_quad(quad, width=85.60):
    edges = [quad[(i + 1) % 4] - quad[i] for i in range(4)]
    lengths = [math.hypot(e[0], e[1]) for e in edges]
    max_idx = int(np.argmax(lengths))
    max_len = lengths[max_idx]
    if max_len < 1e-6:
        return None

    vec = edges[max_idx]
    theta = math.atan2(vec[1], vec[0])
    rot = np.array([
        [ math.cos(theta), math.sin(theta)],
        [-math.sin(theta), math.cos(theta)]
    ])
    scale = width / max_len
    centered = quad - quad.mean(axis=0)
    return (centered @ rot.T) * scale


def quad_score(quad):
    norm = normalize_quad(quad)
    if norm is None:
        return float("inf")
    dist = np.hypot(norm[:, None, 0] - REFERENCE_CARD[None, :, 0],
                    norm[:, None, 1] - REFERENCE_CARD[None, :, 1])
    return (dist.min(axis=1).mean() + dist.min(axis=0).mean()) / 2


camera = cv2.VideoCapture(0)

while True:
    (ret, frame) = camera.read()

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    frame = cv2.Canny(frame, 50, 150)

    lines = cv2.HoughLinesP(frame, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=60)
    quads = find_candidate_quads(lines)

    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line
            cv2.line(frame, (x1, y1), (x2, y2), (60, 60, 60), 1)

    for q in quads:
        cv2.drawContours(frame, [q.reshape(-1, 1, 2)], 0, (0, 255, 0), 1)

    if quads:
        scored = sorted(((quad_score(q), q) for q in quads), key=lambda x: x[0])
        best_score, best_quad = scored[0]
        cv2.drawContours(frame, [best_quad.reshape(-1, 1, 2)], 0, (0, 0, 255), 3)

    cv2.imshow("OpenIPD Alpha", frame)
    if cv2.waitKey(1) == 27:
        break

