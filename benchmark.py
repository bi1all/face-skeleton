import timeit
import numpy as np

# Mock cv2 to avoid dependency issues if any, or just use real cv2
import cv2

def draw_connections_old(canvas, pts, connections, color, thickness=1):
    for a, b in connections:
        if a < len(pts) and b < len(pts):
            cv2.line(canvas,
                     (pts[a][0], pts[a][1]),
                     (pts[b][0], pts[b][1]),
                     color, thickness, cv2.LINE_AA)

def draw_connections_new(canvas, pts, connections, color, thickness=1):
    n = len(pts)
    for a, b in connections:
        if a < n and b < n:
            cv2.line(canvas,
                     (pts[a][0], pts[a][1]),
                     (pts[b][0], pts[b][1]),
                     color, thickness, cv2.LINE_AA)

# Generate dummy data
# typical pts size: 478
pts = [(np.random.randint(0, 1000), np.random.randint(0, 1000), 0) for _ in range(478)]

# typical connections length: ~1300 for tesselation
connections = frozenset((np.random.randint(0, 478), np.random.randint(0, 478)) for _ in range(1300))
color = (255, 255, 255)
canvas1 = np.zeros((1000, 1000, 3), dtype=np.uint8)
canvas2 = np.zeros((1000, 1000, 3), dtype=np.uint8)

setup_code = "from __main__ import draw_connections_old, draw_connections_new, canvas1, canvas2, pts, connections, color"

old_time = timeit.timeit("draw_connections_old(canvas1, pts, connections, color)", setup=setup_code, number=1000)
new_time = timeit.timeit("draw_connections_new(canvas2, pts, connections, color)", setup=setup_code, number=1000)

print(f"Old time (1000 iterations): {old_time:.5f} s")
print(f"New time (1000 iterations): {new_time:.5f} s")
if old_time > 0:
    improvement = (old_time - new_time) / old_time * 100
    print(f"Improvement: {improvement:.2f}%")
