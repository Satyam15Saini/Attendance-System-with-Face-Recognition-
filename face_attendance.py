import cv2
import face_recognition
import pandas as pd
from datetime import datetime, timedelta
import os
import threading
import numpy as np

# Configuration
KNOWN_FACES_DIR = "C:/Users/acer/AI_Attendance_System/known_faces"
CSV_PATH = "attendance.csv"
TOLERANCE = 0.5  # Lower tolerance for stricter matching
DATE_FORMAT = "%d-%m-%Y"
TIME_FORMAT = "%H:%M:%S"
CONFIDENCE_THRESHOLD = 4  # Balanced threshold for stability
IOU_THRESHOLD = 0.4
NOTIFICATION_COOLDOWN = 5  # Seconds between notifications

def calculate_iou(box1, box2):
    """Calculate Intersection over Union between two bounding boxes."""
    top1, right1, bottom1, left1 = box1
    top2, right2, bottom2, left2 = box2

    inter_left = max(left1, left2)
    inter_right = min(right1, right2)
    inter_top = max(top1, top2)
    inter_bottom = min(bottom1, bottom2)

    if inter_right < inter_left or inter_bottom < inter_top:
        return 0.0

    inter_area = (inter_right - inter_left) * (inter_bottom - inter_top)
    area1 = (right1 - left1) * (bottom1 - top1)
    area2 = (right2 - left2) * (bottom2 - top2)
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area != 0 else 0.0

def load_known_faces():
    known_encodings = []
    known_names = []
    
    for file in os.listdir(KNOWN_FACES_DIR):
        image_path = os.path.join(KNOWN_FACES_DIR, file)
        image = face_recognition.load_image_file(image_path)
        
        # Detect faces with higher accuracy for registration
        face_locations = face_recognition.face_locations(image, model="cnn", number_of_times_to_upsample=1)
        encodings = face_recognition.face_encodings(image, face_locations)
        
        if encodings:
            known_encodings.append(encodings[0])
            known_names.append(os.path.splitext(file)[0])
    
    return known_encodings, known_names

class FaceRegister:
    def __init__(self, reload_callback):  # Added reload callback
        self.registering = False
        self.face_image = None
        self.reload_callback = reload_callback  # Store the callback
    
    def register_thread(self):
        name = input("Enter name for new user: ").strip()
        if name and self.face_image is not None:
            filename = f"{name.replace(' ', '_')}.jpg"
            save_path = os.path.join(KNOWN_FACES_DIR, filename)
            
            # Enhance image quality before saving
            gray = cv2.cvtColor(self.face_image, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            cv2.imwrite(save_path, enhanced)
            
            current_time = datetime.now()
            pd.DataFrame({
                "Name": [name],
                "Date": [current_time.strftime(DATE_FORMAT)],
                "Time": [current_time.strftime(TIME_FORMAT)]
            }).to_csv(CSV_PATH, mode='a', header=False, index=False)
            
            print(f"✅ New user {name} registered and logged!")
         # Reload known faces after registration
        self.reload_callback()
        self.registering = False
        self.face_image = None

def main():
    os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
    if not os.path.exists(CSV_PATH):
        pd.DataFrame(columns=["Name", "Date", "Time"]).to_csv(CSV_PATH, index=False)
    
    known_encodings, known_names = load_known_faces()
    
    def reload_known_faces():
        nonlocal known_encodings, known_names
        known_encodings, known_names = load_known_faces()
        print("Known faces reloaded.")

    video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    register = FaceRegister(reload_known_faces)  # Pass reload callback
    
    tracked_faces = []
    daily_attendance = {}
    last_notification = {}
    prev_detected_users = set()

    # Initialize variables for frame skipping
    frame_skip = 2  # Process every 3rd frame
    frame_count = 0

    try:
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % frame_skip != 0:
                continue  # Skip processing this frame

            # Preprocess frame
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # Optimized face detection
            face_locations = face_recognition.face_locations(
                rgb_small_frame,
                model="hog",  # Faster than CNN but less accurate (switch to "cnn" if you have GPU)
                number_of_times_to_upsample=1
            )
            
            # Only compute encodings for detected faces
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            current_time = datetime.now()
            current_date_str = current_time.strftime(DATE_FORMAT)
            current_names = []
            
            for face_encoding in face_encodings:
                # Use face distance for more accurate matching
                face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                
                if face_distances[best_match_index] < TOLERANCE:
                    current_names.append(known_names[best_match_index])
                else:
                    current_names.append("Unknown")

            # Update tracked faces
            new_tracked = []
            current_positions = []
            newly_checked_in = set()
            current_detected_users = set()

            for (top, right, bottom, left), name, encoding in zip(face_locations, current_names, face_encodings):
                top *= 2; right *= 2; bottom *= 2; left *= 2
                current_box = (top, right, bottom, left)
                current_positions.append(current_box)

                best_match = None
                best_iou = IOU_THRESHOLD
                for idx, t_face in enumerate(tracked_faces):
                    iou = calculate_iou(current_box, t_face['position'])
                    if iou > best_iou:
                        best_iou = iou
                        best_match = idx

                if best_match is not None:
                    t_face = tracked_faces.pop(best_match)
                    if name == t_face['name']:
                        t_face['confidence'] = min(t_face['confidence'] + 1, CONFIDENCE_THRESHOLD)
                    else:
                        t_face['confidence'] = max(t_face['confidence'] - 1, 0)
                        if t_face['confidence'] == 0:
                            t_face['name'] = name
                            t_face['confidence'] = 1
                    t_face['position'] = current_box
                    t_face['encoding'] = encoding
                    new_tracked.append(t_face)
                else:
                    new_tracked.append({
                        'name': name,
                        'position': current_box,
                        'encoding': encoding,
                        'confidence': 1
                    })

            tracked_faces.clear()
            tracked_faces.extend(new_tracked)

            # Process attendance and draw tracked faces
            for t_face in tracked_faces:
                if t_face['confidence'] >= CONFIDENCE_THRESHOLD:
                    name = t_face['name']
                    top, right, bottom, left = t_face['position']

                    color = (0, 0, 255)  # Red for unknown
                    label = "Unknown"
                    text_color = (0, 0, 0)

                    if name != "Unknown":
                        current_detected_users.add(name)
                        if name not in daily_attendance:
                            checkin_time = current_time.strftime(TIME_FORMAT)
                            pd.DataFrame({
                                "Name": [name],
                                "Date": [current_date_str],
                                "Time": [checkin_time]
                            }).to_csv(CSV_PATH, mode='a', header=False, index=False)
                            daily_attendance[name] = (current_date_str, checkin_time)
                            print(f"✅ {name} checked in at {current_date_str} {checkin_time}")
                            newly_checked_in.add(name)
                            last_notification[name] = current_time
                            color = (0, 255, 0)  # Green
                        else:
                            color = (0, 255, 255)  # Yellow
                        label = name

                    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(frame, 
                                (left, bottom - text_size[1] - 10),
                                (left + text_size[0] + 10, bottom),
                                (255, 255, 255), -1)
                    cv2.putText(frame, label, 
                               (left + 5, bottom - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

            # Handle reappearance notifications with cooldown
            for name in current_detected_users:
                if (name in daily_attendance and 
                    name not in newly_checked_in and 
                    name not in prev_detected_users):
                    
                    now = datetime.now()
                    if name not in last_notification or \
                       (now - last_notification[name]).total_seconds() >= NOTIFICATION_COOLDOWN:
                        
                        date_str, time_str = daily_attendance[name]
                        print(f"ℹ️ {name} already checked in at {date_str} {time_str}")
                        last_notification[name] = now

            prev_detected_users = current_detected_users.copy()

            # Handle registration
            key = cv2.waitKey(1)
            if key == ord('n') and not register.registering and face_locations:
                top, right, bottom, left = face_locations[0]
                face_img = cv2.cvtColor(rgb_small_frame[top:bottom, left:right], cv2.COLOR_RGB2BGR)
                register.face_image = cv2.resize(face_img, (0, 0), fx=2, fy=2)
                register.registering = True
                threading.Thread(target=register.register_thread, daemon=True).start()

            # Display UI
            cv2.putText(frame, "Press N: New User", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(frame, "Press Q: Quit", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.imshow('Attendance System', frame)

            if key == ord('q'):
                break

    finally:
        video_capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()