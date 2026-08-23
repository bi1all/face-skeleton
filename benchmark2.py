import timeit
import numpy as np
import cv2

def draw_connections_old(canvas, pts, connections, color, thickness=1):
    for a, b in connections:
        if a < len(pts) and b < len(pts):
            pass

def draw_connections_new(canvas, pts, connections, color, thickness=1):
    n = len(pts)
    for a, b in connections:
        if a < n and b < n:
            pass

# Generate dummy data
pts = [(0, 0, 0) for _ in range(478)]
connections = frozenset((np.random.randint(0, 478), np.random.randint(0, 478)) for _ in range(1300))
color = (255, 255, 255)
canvas = np.zeros((1000, 1000, 3), dtype=np.uint8)

setup_code = "from __main__ import draw_connections_old, draw_connections_new, canvas, pts, connections, color"

old_time = timeit.timeit("draw_connections_old(canvas, pts, connections, color)", setup=setup_code, number=10000)
new_time = timeit.timeit("draw_connections_new(canvas, pts, connections, color)", setup=setup_code, number=10000)

print(f"Loop overhead - Old time (10000 iterations): {old_time:.5f} s")
print(f"Loop overhead - New time (10000 iterations): {new_time:.5f} s")
if old_time > 0:
    improvement = (old_time - new_time) / old_time * 100
    print(f"Improvement: {improvement:.2f}%")
