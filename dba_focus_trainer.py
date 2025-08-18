import argparse
import cv2
import numpy as np
import pyautogui
import time
import mss
import pygetwindow as gw

# Globals for mouse callback
drawing = False
ix, iy = -1, -1
rx, ry, rw, rh = 0, 0, 0, 0
region_selected = False

def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, rx, ry, rw, rh, region_selected, preview_img, preview_img_copy

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        region_selected = False

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            preview_img = preview_img_copy.copy()
            cv2.rectangle(preview_img, (ix, iy), (x, y), (0, 0, 255), 2)
            cv2.imshow("Select Capture Region", preview_img)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        rx, ry = min(ix, x), min(iy, y)
        rw, rh = abs(x - ix), abs(y - iy)
        region_selected = True
        preview_img = preview_img_copy.copy()
        cv2.rectangle(preview_img, (rx, ry), (rx + rw, ry + rh), (0, 0, 255), 2)
        cv2.imshow("Select Capture Region", preview_img)

def non_max_suppression(boxes, scores, overlapThresh):
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    scores = np.array(scores)

    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,0] + boxes[:,2]
    y2 = boxes[:,1] + boxes[:,3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= overlapThresh)[0]
        order = order[inds + 1]

    return keep


def images_different(img1, img2, gray_match_threshold=1200, color_diff_threshold=300):
    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Grayscale comparison
    gray_diff = cv2.absdiff(gray1, gray2)
    gray_diff_count = np.sum(gray_diff > 30)

    # Color comparison
    color_diff = cv2.absdiff(img1, img2)
    color_diff_count = np.sum(color_diff > 30)

    # Conditions
    same_shape = gray_diff_count < gray_match_threshold
    color_changed = color_diff_count > color_diff_threshold

    return same_shape and color_changed


def focus_dream_seek_window():
     # Focus Dream Seek window
        app_windows = [w for w in gw.getWindowsWithTitle("Dragonball Asylum") if w.title]
        if not app_windows:
            print("ERROR: Dragonball Asylum not found. Please open the app first.")
            exit()

        dream_seek_window = app_windows[0]
        dream_seek_window.activate()
        time.sleep(1)

        print("Dream Seek window focused.")
        return dream_seek_window

def restart_focus_train():
    pyautogui.press("k")
    time.sleep(1)
    pyautogui.press("enter")

def main(in_delay):
    global preview_img, preview_img_copy, rx, ry, rw, rh, region_selected, action_delay
    
    action_delay = in_delay
    MAX_DELAY = 2.0 # seconds

    # Step 1: Focus Dream Seek window
    dream_seek_window = focus_dream_seek_window()

    # Window region
    window_left = dream_seek_window.left
    window_top = dream_seek_window.top
    window_width = dream_seek_window.width
    window_height = dream_seek_window.height

    initial_region = {
        "top": window_top,
        "left": window_left,
        "width": window_width,
        "height": window_height
    }

    print(f"Full window region: {initial_region}")

    sct = mss.mss()
    screenshot = np.array(sct.grab(initial_region))
    preview_img = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
    preview_img_copy = preview_img.copy()

    cv2.namedWindow("Select Capture Region")
    cv2.setMouseCallback("Select Capture Region", draw_rectangle)

    print("INPUT REQUIRED!! Draw a rectangle with the mouse to select capture region.")
    print("Press ENTER to confirm selection, or ESC to exit.")

    while True:
        cv2.imshow("Select Capture Region", preview_img)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            print("Exiting...")
            cv2.destroyAllWindows()
            exit()

        if key == 13:
            if region_selected and rw > 5 and rh > 5:
                print(f"Capture region selected at (x={rx}, y={ry}, w={rw}, h={rh})")
                focus_dream_seek_window()
                break
            else:
                print("WARNING: Please draw a valid region before pressing ENTER.")

    cv2.destroyAllWindows()

    capture_region = {
        "top": window_top + ry,
        "left": window_left + rx,
        "width": rw,
        "height": rh
    }
    print(f"Final capture region on screen: {capture_region}")

    templates = {
        "left": cv2.imread("FCC/arrow_left.png"),
        "right": cv2.imread("FCC/arrow_right.png"),
        "up": cv2.imread("FCC/arrow_up.png"),
        "down": cv2.imread("FCC/arrow_down.png")
    }

    threshold = 0.7
    while True:
        screenshot = np.array(sct.grab(capture_region))
        screenshot_bgr = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)

        detected_boxes = []
        detected_scores = []
        detected_directions = []

        for direction, template in templates.items():
            if template is None:
                print(f"WARNING: Missing template arrow-{direction}.png")
                continue

            w, h = template.shape[1], template.shape[0]
            res = cv2.matchTemplate(screenshot_bgr, template, cv2.TM_CCOEFF_NORMED)

            mask = (res >= threshold).astype(np.uint8)
            kernel = np.ones((3,3), np.uint8)
            dilated = cv2.dilate(res, kernel)
            local_max = (res == dilated)
            peaks = np.where(mask & local_max)

            for pt in zip(*peaks[::-1]):
                score = res[pt[1], pt[0]]
                detected_boxes.append([pt[0], pt[1], w, h])
                detected_scores.append(score)
                detected_directions.append(direction)

        keep_indices = non_max_suppression(detected_boxes, detected_scores, 0.5)

        detected_arrows = []
        for i in keep_indices:
            x, y, w, h = detected_boxes[i]
            direction = detected_directions[i]
            detected_arrows.append((x, y, direction))
            cv2.rectangle(screenshot_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)

        detected_arrows.sort(key=lambda x: x[0])  # sort left to right

        if not detected_arrows:
            print("WARNING: No arrows detected.")
            restart_focus_train()
            action_delay += 0.1
            if action_delay > MAX_DELAY:
                action_delay = MAX_DELAY
            continue

        max_arrows = 5
        for idx, (x, y, direction) in enumerate(detected_arrows, start=1):
            w, h = templates[direction].shape[1], templates[direction].shape[0]

            # Crop before pressing key
            before_arrow = screenshot_bgr[y:y+h, x:x+w]

            # Press the key
            pyautogui.press(direction)
            time.sleep(action_delay)

            # Capture again and crop
            after_screenshot = np.array(sct.grab(capture_region))
            after_screenshot_bgr = cv2.cvtColor(after_screenshot, cv2.COLOR_BGRA2BGR)
            after_arrow = after_screenshot_bgr[y:y+h, x:x+w]

            num_retries = 3

            # If we're on the 5th arrow, skip validation and restart detection
            if idx == max_arrows:
                print(f"Input for 5th arrow ({direction}), skipping validation and restarting detection.")
                time.sleep(action_delay)
                continue

            while (num_retries > 0 and not images_different(before_arrow, after_arrow)):
                print(f"WARNING: Arrow {direction} still looks the same, waiting")
                time.sleep(action_delay)
                after_screenshot = np.array(sct.grab(capture_region))
                after_screenshot_bgr = cv2.cvtColor(after_screenshot, cv2.COLOR_BGRA2BGR)
                after_arrow = after_screenshot_bgr[y:y+h, x:x+w]
                num_retries -= 1

            if num_retries == 0:
                print(f"ERROR: Arrow {direction} did not change after retries, skipping")
                break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arrow recognition script for DBA")
    parser.add_argument("--delay", type=float, default=0.1,
                        help="Delay (in seconds) after each key press before checking arrows")
    args = parser.parse_args()

    print(f"[CONFIG] Action delay set to {args.delay} seconds")
    main(args.delay)

