import cv2
import os
import requests
import math
import cvzone
from ultralytics import YOLO

# --- Configuration ---
VIDEO_URL = "https://raw.githubusercontent.com/wdzhong/traffic-video-process/master/surveillance.m4v" 
VIDEO_PATH = r"d:\TrafficProject\data\surveillance.mp4"

def download_video(url, save_path):
    """Downloads a sample video if it doesn't exist."""
    if os.path.exists(save_path):
        print(f"Video found at {save_path}")
        return

    print(f"Downloading sample video from {url}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status() 
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete!")
    except Exception as e:
        print(f"Error downloading video: {e}")
        print("Please manually add a 'traffic.mp4' file to the data folder.")

def main():
    # 1. Setup Data
    download_video(VIDEO_URL, VIDEO_PATH)

    # 2. Open Video Capture
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print(f"Error: Could not open video file {VIDEO_PATH}")
        return

    # Load the AI Model (Back to 'Medium' as per user request)
    print("Loading AI Model (Medium Version)...")
    model = YOLO('yolov8m.pt') 
    
    # Class Names
    classNames = model.names

    print("Starting Video Stream... Press 'q' to exit.")

    while True:
        success, frame = cap.read()
        
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame = cv2.resize(frame, (1280, 720))

        # --- AI DETECTION START ---
        # conf=0.25: Standard threshold (Balanced)
        results = model(frame, stream=True, conf=0.25)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                # 1. Get Box Coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1

                # 2. Get Class & Confidence
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                currentClass = classNames[cls]

                # 3. Filter for Vehicles Only
                if currentClass in ["car", "truck", "bus", "motorbike", "bicycle"]:
                    
                    # Set Color based on type (BGR Format)
                    if currentClass in ["motorbike", "bicycle"]:
                        color = (0, 255, 255) # Yellow for bikes (High visibility)
                    elif currentClass in ["bus", "truck"]:
                        color = (0, 165, 255) # Orange for heavy vehicles
                    else:
                        color = (0, 255, 0)   # Green for cars

                    # Draw Thin, Clean Rectangle
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2) 

                    # Clean Label
                    label = f'{currentClass} {conf}'
                    t_size = cv2.getTextSize(label, 0, fontScale=0.6, thickness=1)[0]
                    c2 = x1 + t_size[0], y1 - t_size[1] - 3
                    
                    # Draw small background for text (for readability)
                    cv2.rectangle(frame, (x1, y1), c2, color, -1, cv2.LINE_AA) 
                    # Draw text
                    cv2.putText(frame, label, (x1, y1 - 2), 0, 0.6, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

        # --- AI DETECTION END ---
        
        # Display the frame
        cv2.imshow("Traffic Monitor System", frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
