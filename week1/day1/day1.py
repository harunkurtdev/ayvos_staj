import cv2
from ultralytics import YOLO
from ultralytics.engine.results import Boxes
from deep_sort_realtime.deepsort_tracker import DeepSort,Tracker

model:YOLO = YOLO("yolov8n.pt")

tracker = DeepSort(max_age=50)

cap = cv2.VideoCapture(1)

count = 0
counted_ids = set()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    results = model.predict(frame, classes=0, conf=0.8)[0]
    
        
    detections = [(det.boxes.xyxy[0], det.boxes.conf, det.boxes.cls) for det in results]
    
    tracks = tracker.update_tracks(detections, frame=frame)
    
    for track in tracks:
        if track.is_confirmed() and track.track_id not in counted_ids:
            counted_ids.add(track.track_id)
            count += 1
            track_id = track.track_id
            bbox = track.to_ltwh()  

            x, y, w, h = map(int, bbox)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            cv2.putText(frame, f"ID: {track_id}", (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.putText(frame, f"Count: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(25) == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()
