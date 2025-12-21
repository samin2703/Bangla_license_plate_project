import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np

# load model once
model = YOLO("best.pt")


def group_boxes_into_lines(result):
    """Return list of text lines from YOLO result."""
    detections = []

    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
        cls_id = int(box.cls[0])
        label = model.names[cls_id]

        xc = (x1 + x2) / 2
        yc = (y1 + y2) / 2
        h = (y2 - y1)
        detections.append({"xc": xc, "yc": yc, "h": h, "label": label})

    if not detections:
        return []

    # sort vertically
    detections.sort(key=lambda d: d["yc"])
    lines = [[detections[0]]]

    for det in detections[1:]:
        current_line = lines[-1]
        avg_y = np.mean([d["yc"] for d in current_line])
        avg_h = np.mean([d["h"] for d in current_line])

        if abs(det["yc"] - avg_y) < avg_h:
            current_line.append(det)
        else:
            lines.append([det])

    plate_lines = []
    for line in lines:
        line.sort(key=lambda d: d["xc"])  # left to right
        labels = [d["label"] for d in line]
        if all(ch.isdigit() for ch in "".join(labels)):
            text_line = "".join(labels)   # number row
        else:
            text_line = " ".join(labels)  # text row
        plate_lines.append(text_line)

    return plate_lines


st.title("Bangla License Plate Reader")

uploaded_file = st.file_uploader("Upload license plate image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # read image into OpenCV
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # run model
    results = model.predict(source=img, conf=0.25)
    r = results[0]

    # annotated image
    annotated = r.plot()
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    # extract text lines
    plate_lines = group_boxes_into_lines(r)

    st.image(annotated_rgb, caption="Predictions", use_container_width=True)

    if plate_lines:
        st.subheader("Recognized Text")
        for line in plate_lines:
            st.write(line)
    else:
        st.write("No characters detected.")