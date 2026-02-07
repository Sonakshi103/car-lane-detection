# Lane Detection Accuracy Evaluation + UI Dashboard (Streamlit)

"""
This file contains:
✅ TuSimple lane accuracy calculation
✅ Streamlit UI dashboard to upload/view results

Run UI using:
    streamlit run lane_accuracy_ui.py
"""

import os
import json
import cv2
import numpy as np
from sklearn.linear_model import LinearRegression
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# ✅ Load TuSimple Annotations
# -----------------------------
def load_annotations(json_file):
    """Load TuSimple annotation JSON from Streamlit upload"""
    import json
    data = [json.loads(line) for line in json_file.read().decode('utf-8').splitlines()]
    return data



# -----------------------------
# ✅ Compute Lane Accuracy (TuSimple format)
# -----------------------------
def compute_lane_accuracy(pred_lanes, gt_lanes, threshold=20):
    total_points = 0
    matched_points = 0

    for pred, gt in zip(pred_lanes, gt_lanes):
        for px, gx in zip(pred, gt):
            if gx == -2:  # ground truth missing
                continue
            total_points += 1
            if abs(px - gx) <= threshold:
                matched_points += 1
    if total_points == 0:
        return 0
    return matched_points / total_points

# -----------------------------
# ✅ Lane Regression Model
# -----------------------------
def predict_lane(h_samples, lane_points):
    pts = [(x,y) for x,y in zip(lane_points, h_samples) if x != -2]
    if len(pts) < 2:
        return [-2]*len(h_samples)
    ys = np.array([y for _,y in pts]).reshape(-1,1)
    xs = np.array([x for x,_ in pts])
    model = LinearRegression().fit(ys,xs)
    pred = model.predict(np.array(h_samples).reshape(-1,1))
    return pred.astype(int).tolist()

# -----------------------------
# ✅ Streamlit UI
# -----------------------------
st.title("🚗 Lane Detection Accuracy Dashboard")
st.write("Upload TuSimple annotation file to compute accuracy.")

uploaded_json = st.file_uploader("Upload TuSimple JSON file", type=["json"])
img_dir = st.text_input("Enter image directory path for visualization (optional)")

if uploaded_json:
    data = load_annotations(uploaded_json)
    st.success(f"Loaded {len(data)} samples ✔️")

    results = []
    for sample in data[:50]:  # evaluate first 50 samples
        h = sample.get('h_samples') or sample.get('h_sample')
        gt_lanes = sample['lanes']
        pred_lanes = [predict_lane(h, lane) for lane in gt_lanes]
        acc = compute_lane_accuracy(pred_lanes, gt_lanes)
        results.append(acc)

    avg_acc = np.mean(results)
    st.metric("Average Lane Accuracy", f"{avg_acc*100:.2f}%")

    df = pd.DataFrame({"Frame": list(range(len(results))), "Accuracy": results})
    st.line_chart(df, x="Frame", y="Accuracy")

    if img_dir:
        sample = data[0]
        img_path = os.path.join(img_dir, sample['raw_file'])
        img = cv2.imread(img_path)
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Sample Frame")
