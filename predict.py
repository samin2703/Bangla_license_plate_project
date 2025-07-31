from ultralytics import YOLO
import cv2

model = YOLO("best.pt")  

results = model.predict(source="test_image.jpg", save=True, conf=0.25)

for r in results:
    annotated_frame = r.plot() 
    cv2.imshow("Prediction", annotated_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
