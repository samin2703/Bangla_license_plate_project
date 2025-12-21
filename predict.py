from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("best.pt")

img_path = "test1.png"
results = model.predict(source=img_path, save=True, conf=0.25)

for r in results:
    annotated_frame = r.plot()

    detections = []
    # collect center positions + labels
    for box in r.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cls_id = int(box.cls[0])
        label = model.names[cls_id]

        xc = (x1 + x2) / 2
        yc = (y1 + y2) / 2
        h = (y2 - y1)
        detections.append({"xc": xc, "yc": yc, "h": h, "label": label})

    if not detections:
        continue

    # -------- group detections into text lines (rows) ----------
    detections.sort(key=lambda d: d["yc"])  # sort by vertical position
    lines = [[detections[0]]]

    for det in detections[1:]:
        current_line = lines[-1]
        avg_y = np.mean([d["yc"] for d in current_line])
        avg_h = np.mean([d["h"] for d in current_line])

        # same row if y-distance smaller than avg box height of the row
        if abs(det["yc"] - avg_y) < avg_h:
            current_line.append(det)
        else:
            lines.append([det])

    # -------- sort each line left‑to‑right and build strings ----------
    plate_lines = []
    for line in lines:
        line.sort(key=lambda d: d["xc"])      # left to right
        labels = [d["label"] for d in line]

        # if the line is all digits, join without spaces (e.g. 150568)
        if all(ch.isdigit() for ch in "".join(labels)):
            text_line = "".join(labels)
        else:
            text_line = " ".join(labels)

        plate_lines.append(text_line)

    plate_text = "\n".join(plate_lines)

    # print in terminal
    print("Predicted plate:")
    print(plate_text)        # e.g.:
                             # Dhaka Metro Ga
                             # 150568

    # draw text on image
    y0 = 30
    for i, text_line in enumerate(plate_lines):
        y = y0 + i * 30
        cv2.putText(
            annotated_frame,
            text_line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    cv2.imshow("Prediction", annotated_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()