import datetime
import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


from helper import create_video_writer

model = YOLO("yolov8n.pt")

tracker = DeepSort(
    max_age=30,
    max_cosine_distance=0.2, 
    nms_max_overlap=1.0,
    nn_budget=100
)


cap = cv2.VideoCapture("test.mp4") 

polygons = [[]]
polygon_counts = [0]
counted_ids_per_polygon = [set()]
current_polygon = 0  

previous_positions = {}

writer = create_video_writer(cap, "predict_test_video.mp4")

paused = False

polygon_entries = [0]  
polygon_exits = [0] 

detection_enabled = False


def get_polygon_center(polygon):
    if len(polygon) < 3:
        return None  
    x_coords = [p[0] for p in polygon]
    y_coords = [p[1] for p in polygon]
    return (sum(x_coords) // len(x_coords), sum(y_coords) // len(y_coords))

# Fare ile poligon çizme fonksiyonu
def draw_polygon(event, x, y, flags, param):
    global polygons, polygon_counts, counted_ids_per_polygon, current_polygon
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # Eğer mevcut poligon yoksa, yeni bir tane oluştur.
        while len(polygons) <= current_polygon:
            polygons.append([])  
            polygon_counts.append(0)
            counted_ids_per_polygon.append(set())

        # Tıklanan noktayı ekle
        polygons[current_polygon].append((x, y))

    elif event == cv2.EVENT_RBUTTONDOWN:
        # Eğer mevcut poligon varsa sil
        if polygons and current_polygon < len(polygons):
            polygons.pop()
            polygon_counts.pop()
            counted_ids_per_polygon.pop()
            current_polygon = max(0, current_polygon - 1)
SCALE_FACTOR = 1

cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", draw_polygon)

while cap.isOpened():
    start = datetime.datetime.now()

    if not paused:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0) 
            continue
            
        for poly in polygons:
            if len(poly) > 1:
                cv2.polylines(frame, [np.array(poly, np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)

        
        if detection_enabled:
            results = model(frame)[0]
            detections = []

            for result in results.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = result
                if int(class_id) == 0 and score > 0.5:
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    width = x2 - x1
                    height = y2 - y1
                    # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                         
                    center_x = (x1 + x2) // 2
                    center_y = (y2)
                    
                    in_polygon = False
                    for poly in polygons:
                        if len(poly) >= 3:
                            pts = np.array(poly, np.int32)
                            if cv2.pointPolygonTest(pts, (center_x, center_y), False) >= 0:
                                in_polygon = True
                                break
                    if in_polygon:

                        detections.append(([x1, y1, x2, y2], score, "person"))

            tracks = tracker.update_tracks(detections, frame=frame)

            for track in tracks:
                if not track.is_confirmed():
                    continue

                track_id = track.track_id
                ltrb = track.to_ltwh()
                x1, y1, x2, y2 = map(int, ltrb)

                center_x, center_y = (x1 + x2) // 2, ( y2) 

                if track_id in previous_positions:
                    prev_x, prev_y = previous_positions[track_id]
                    cv2.arrowedLine(frame, (prev_x, prev_y), (center_x, center_y), (255, 0, 0), 2, tipLength=0.3)

                

                for i, poly in enumerate(polygons):
                    if len(poly) > 1:

                        polygon_center = get_polygon_center(poly)  # Poligon merkezini al
                        
                        if polygon_center is None:
                            continue

                        poly_x, poly_y = polygon_center

                        polygon_np = np.array(poly, dtype=np.int32)
                        if cv2.pointPolygonTest(polygon_np, (center_x, center_y), False) >= 0:
                            if track_id not in counted_ids_per_polygon[i]:
                                counted_ids_per_polygon[i].add(track_id)
                                polygon_counts[i] += 1
                            
                            if track_id in previous_positions:
                                prev_x, prev_y = previous_positions[track_id]
                                if prev_y >= poly_y and center_y < poly_y:
                                    polygon_exits[i] += 1
                                elif prev_y <= poly_y and center_y > poly_y:
                                    polygon_entries[i] += 1

                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            
                            previous_positions[track_id] = (center_x, center_y)
                            break
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    end = datetime.datetime.now()
    fps = f"FPS: {1 / (end - start).total_seconds():.2f}"

    (text_width, text_height), _ = cv2.getTextSize(fps, cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
    fps_x = frame.shape[1] - text_width - 50  
    fps_y = 50  
    cv2.putText(frame, fps, (fps_x, fps_y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

    for i in range(len(polygons)):
        cv2.putText(frame, f"Poly {i+1} Giren: {polygon_entries[i]} Cikan: {polygon_exits[i]}", 
                    (10, 60 + (i+1) * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow("Frame", frame)
    writer.write(frame)

    key = cv2.waitKey(30) & 0xFF

    if key == ord("q"):
        break
    elif key == ord("a"):
        polygons.append([])
        polygon_counts.append(0)
        polygon_entries.append(0)  
        polygon_exits.append(0)  
        counted_ids_per_polygon.append(set())
        
        current_polygon += 1
    elif key == ord("z"):
        # En son yapılan işlemi geri al
        if polygons[current_polygon]:
            polygons[current_polygon].pop() 
        elif current_polygon > 0:
            polygons.pop()  
            polygon_counts.pop()
            polygon_entries.pop()  
            polygon_exits.pop() 
            counted_ids_per_polygon.pop()
            current_polygon -= 1 
    elif key == ord(" "): 
        paused = not paused
    elif key == ord("d"):  
        detection_enabled = not detection_enabled

cap.release()
writer.release()
cv2.destroyAllWindows()

