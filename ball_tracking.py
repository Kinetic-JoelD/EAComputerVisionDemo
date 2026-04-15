from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64, help="max buffer size")
args = vars(ap.parse_args())

color_ranges = {
    "red":    ((170, 86,  6),  (180, 255, 255)),
    "orange": ((0,   86,  6),  (25,  255, 255)),
    "yellow": ((20,  86,  86), (30,  255, 255)),
    "green":  ((29,  86,  6),  (80,  255, 255)),
    "blue":   ((80,  86,  6),  (135, 255, 255)),
    "purple": ((135, 86,  6),  (170, 255, 255)),
}

# BGR draw colors for each tracked color
draw_colors = {
    "red":    (0,   0,   255),
    "orange": (0,   165, 255),
    "yellow": (0,   255, 255),
    "green":  (0,   255, 0),
    "blue":   (255, 0,   0),
    "purple": (128, 0,   128),
}

# Which colors are actively tracked (toggle with keys)
active_colors = {"green"}

# Separate trail buffer for each color
pts = {color: deque(maxlen=args["buffer"]) for color in color_ranges}

if not args.get("video", False):
    vs = VideoStream(src=0).start()
else:
    vs = cv2.VideoCapture(args["video"])
time.sleep(2.0)

# Key bindings to toggle each color on/off
key_bindings = {
    ord("1"): "green",
    ord("2"): "red",
    ord("3"): "orange",
    ord("4"): "blue",
    ord("5"): "yellow",
    ord("6"): "purple",
}

while True:
    frame = vs.read()
    frame = frame[1] if args.get("video", False) else frame
    if frame is None:
        break

    frame = cv2.flip(frame, 1)
    frame = imutils.resize(frame, width=600)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # --- Loop over every active color ---
    for color in active_colors:
        lower, upper = color_ranges[color]
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        center = None

        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            if radius > 10:
                cv2.circle(frame, (int(x), int(y)), int(radius), draw_colors[color], 2)
                cv2.circle(frame, center, 5, draw_colors[color], -1)
                # Label the detected object
                cv2.putText(frame, color, (int(x) - 10, int(y) - int(radius) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, draw_colors[color], 2)

        pts[color].appendleft(center)

        # Draw trail for this color
        for i in range(1, len(pts[color])):
            if pts[color][i - 1] is None or pts[color][i] is None:
                continue
            thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
            cv2.line(frame, pts[color][i - 1], pts[color][i], draw_colors[color], thickness)

    # HUD: show which colors are active
    hud_y = 20
    for color, bgr in draw_colors.items():
        status = "ON" if color in active_colors else "off"
        cv2.putText(frame, f"{color}: {status}", (10, hud_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr if status == "ON" else (100, 100, 100), 1)
        hud_y += 18

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    # Toggle colors on/off with number keys
    if key in key_bindings:
        color = key_bindings[key]
        if color in active_colors:
            active_colors.remove(color)
            pts[color].clear()
            print(f"Stopped tracking {color.upper()}")
        else:
            active_colors.add(color)
            print(f"Now tracking {color.upper()}")

if not args.get("video", False):
    vs.stop()
else:
    vs.release()
cv2.destroyAllWindows()