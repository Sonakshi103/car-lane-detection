# Streamlit UI — Image-by-Image Lane Detection + YOLO (yolov8s)
"""
Usage:
    python -m streamlit run lane_full_ui.py

What this UI does (image mode):
- Upload a single image (or choose sample from dataset)
- Optional: upload corresponding TuSimple JSON line to draw GT lanes
- Detect lanes using either TuSimple GT (if provided) or Hough-based fallback
- Draw regression-smoothed lanes (using user's LinearRegression approach)
- Run YOLOv8s object detection and show object count + boxes
- Show lane-deviation status and timestamp
- Save output image to results/

Notes:
- The UI tries to load a local 'yolov8s.pt' model file. If not found, it falls back to the remote 'yolov8s' model name (which will download automatically via ultralytics).
- Dependencies: streamlit, ultralytics, opencv-python, numpy, scikit-learn, matplotlib, pillow
"""

import os
import io
import json
import time
from datetime import datetime

import cv2
import numpy as np
from sklearn.linear_model import LinearRegression
from ultralytics import YOLO
import streamlit as st
from PIL import Image

# -----------------------------
# Helper functions (adapted from your pipeline)
# -----------------------------

def enhance_night_mode(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.equalizeHist(v)
    enhanced = cv2.merge((h, s, v))
    return cv2.cvtColor(enhanced, cv2.COLOR_HSV2BGR)


def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    return edges


def region_of_interest(img):
    height, width = img.shape
    mask = np.zeros_like(img)
    polygon = np.array([[
        (0, height),
        (width, height),
        (int(0.6 * width), int(0.5 * height)),
        (int(0.4 * width), int(0.5 * height)),
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    masked = cv2.bitwise_and(img, mask)
    return masked


def draw_regression_lanes(image, lanes, h_samples):
    """Draw smooth regression lines given lanes (list of x lists)."""
    output = image.copy()
    for lane in lanes:
        points = [(x, y) for x, y in zip(lane, h_samples) if x != -2]
        if len(points) > 2:
            ys = np.array([y for _, y in points]).reshape(-1, 1)
            xs = np.array([x for x, _ in points])
            model = LinearRegression().fit(ys, xs)
            y_pred = np.array(h_samples).reshape(-1, 1)
            x_pred = model.predict(y_pred).astype(np.int32)
            for i in range(len(h_samples) - 1):
                pt1 = (int(x_pred[i]), int(h_samples[i]))
                pt2 = (int(x_pred[i+1]), int(h_samples[i+1]))
                cv2.line(output, pt1, pt2, (0, 255, 0), 3)
    return output


def check_lane_deviation(image, lanes):
    height, width, _ = image.shape
    mid = width // 2
    lane_points = [x for lane in lanes for x in lane if x != -2]
    if not lane_points:
        return "⚠️ Lanes Not Detected"
    avg_lane_center = np.mean([min(lane_points), max(lane_points)])
    deviation = mid - avg_lane_center
    if abs(deviation) > 80:
        return "🚨 Lane Deviation Alert!"
    else:
        return "✅ Lane Centered"


# -----------------------------
# Fallback lane estimator using Hough (if no TuSimple GT provided)
# -----------------------------

def hough_lane_estimate(image, h_samples=None):
    """Estimate left/right lane x-coordinates at h_samples using Hough lines.
    Returns lanes as a list: [left_lane_xs, right_lane_xs]
    If cannot find, returns [[-2...], [-2...]] matching length of h_samples.
    """
    edges = preprocess_image(image)
    roi = region_of_interest(edges)
    lines = cv2.HoughLinesP(roi, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=50)

    height, width = image.shape[:2]
    if h_samples is None:
        # default sample y-coordinates
        h_samples = list(range(height - 1, int(height * 0.3), -20))
    left_lines = []
    right_lines = []
    if lines is not None:
        for l in lines:
            x1, y1, x2, y2 = l[0]
            slope = (y2 - y1) / (x2 - x1 + 1e-6)
            if abs(slope) < 0.3:
                continue
            if slope < 0:
                left_lines.append((x1, y1, x2, y2))
            else:
                right_lines.append((x1, y1, x2, y2))
    def fit_lines(line_list):
        if not line_list:
            return [-2] * len(h_samples)
        xs = []
        ys = []
        for x1, y1, x2, y2 in line_list:
            xs += [x1, x2]
            ys += [y1, y2]
        if len(xs) < 2:
            return [-2] * len(h_samples)
        model = LinearRegression().fit(np.array(ys).reshape(-1, 1), np.array(xs))
        preds = model.predict(np.array(h_samples).reshape(-1, 1)).astype(int).tolist()
        return preds

    left_xs = fit_lines(left_lines)
    right_xs = fit_lines(right_lines)
    return [left_xs, right_xs], h_samples


# -----------------------------
# YOLO model loader
# -----------------------------

def load_yolo_model(preferred_name="yolov8s.pt"):
    try:
        model = YOLO(preferred_name)
        return model
    except Exception:
        try:
            # fallback to model name which ultralytics can download
            base = preferred_name.replace('.pt', '')
            model = YOLO(base)
            return model
        except Exception as e:
            st.error(f"Failed to load YOLO model: {e}")
            return None


# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(page_title="Lane Detection + YOLO UI", layout="wide")
st.title("🚗 Lane Detection + YOLO — Image Mode")

with st.sidebar:
    st.header("Settings")
    model_choice = st.selectbox("YOLO model to use", ["yolov8s.pt", "yolov8n.pt", "yolov8m.pt"]) 
    night_mode = st.checkbox("Apply night enhancement", value=True)
    save_results = st.checkbox("Save output image to results/", value=True)
    threshold_deviation = st.slider("Lane deviation threshold (px)", 20, 200, 80)

st.info("Upload an image. Optionally upload a single-line TuSimple JSON file for GT lanes (useful for visual comparison). If no JSON is provided, the UI tries a Hough-based lane estimate.")

col1, col2 = st.columns([1, 1])

with col1:
    uploaded_img = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
    uploaded_json = st.file_uploader("Optional: upload TuSimple JSON line for this image", type=["json", "txt"])
    run_button = st.button("Process Image")

with col2:
    st.markdown("**Preview / Output**")
    output_placeholder = st.empty()
    stats_placeholder = st.empty()

# Load model when needed
yolo_model = None
if run_button:
    if uploaded_img is None:
        st.warning("Please upload an image to process.")
    else:
        # read image
        image_bytes = uploaded_img.read()
        image_pil = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

        if night_mode:
            image = enhance_night_mode(image)

        # Get lanes from GT JSON if provided
        lanes = None
        h_samples = None
        if uploaded_json is not None:
            try:
                txt = uploaded_json.read().decode('utf-8').strip()
                # If uploaded file contains multiple lines, try to find a line with matching raw_file filename
                # But for simplicity assume it's one-line JSON for the current image
                sample = json.loads(txt)
                lanes = sample.get('lanes')
                h_samples = sample.get('h_samples') or sample.get('h_sample')
            except Exception as e:
                st.warning(f"Failed to parse uploaded JSON: {e}")

        # If no GT lanes provided, estimate using Hough
        if lanes is None:
            (lanes, h_samples) = hough_lane_estimate(image)

        # Draw regression lanes
        lane_img = draw_regression_lanes(image, lanes, h_samples)

        # Load YOLO
        if yolo_model is None:
            with st.spinner("Loading YOLO model (this may take a few seconds)..."):
                yolo_model = load_yolo_model(model_choice)

        # Detect objects
        obj_count = 0
        detected_img = lane_img
        if yolo_model is not None:
            try:
                results = yolo_model(detected_img)
                # results[0].plot() returns numpy BGR image with boxes
                detected_img = results[0].plot()
                obj_count = len(results[0].boxes)
            except Exception as e:
                st.warning(f"YOLO detection failed: {e}")

        # Deviation (use same logic but with custom threshold)
        height, width = detected_img.shape[:2]
        mid = width // 2
        lane_points = [x for lane in lanes for x in lane if x != -2]
        if not lane_points:
            status = "⚠️ Lanes Not Detected"
        else:
            avg_lane_center = np.mean([min(lane_points), max(lane_points)])
            deviation = mid - avg_lane_center
            if abs(deviation) > threshold_deviation:
                status = "🚨 Lane Deviation Alert!"
            else:
                status = "✅ Lane Centered"

        # Overlay text
        cv2.putText(detected_img, f"Objects: {obj_count}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(detected_img, status, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(detected_img, datetime.now().strftime("%H:%M:%S"), (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 255), 2)

        # Show output
        rgb_img = cv2.cvtColor(detected_img, cv2.COLOR_BGR2RGB)
        output_placeholder.image(rgb_img, use_column_width=True)

        # Save if requested
        if save_results:
            os.makedirs("results", exist_ok=True)
            name = uploaded_img.name
            out_path = os.path.join("results", f"processed_{name}")
            cv2.imwrite(out_path, detected_img)

        # Stats
        stats_placeholder.markdown(
    f"""
**Objects detected:** {obj_count}  
**Status:** {status}  
**YOLO model:** {model_choice}
"""
)

st.write("---")
st.caption("If you want video or webcam mode, tell me and I'll extend this UI to include real-time processing.")

