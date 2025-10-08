import cv2
import numpy as np
import requests
from flask import Flask, render_template, Response
import threading
import time
from datetime import datetime

app = Flask(__name__)

class ObjectDetector:
    def __init__(self):
        # Load YOLO model
        self.net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
        
        # Load class names
        with open("coco.names", "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        
        # Get output layer names
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        
        # Colors for different classes
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
        
        # ESP32-CAM stream URL
        self.stream_url = "http://192.168.1.100"  # Replace with ESP32-CAM IP
        
        # Detection results
        self.latest_frame = None
        self.detection_results = []
        
    def detect_objects(self, frame):
        """Perform object detection on a frame"""
        height, width, channels = frame.shape
        
        # Prepare the frame for YOLO
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)
        
        # Extract information from outputs
        boxes = []
        confidences = []
        class_ids = []
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > 0.5:  # Confidence threshold
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Apply non-maximum suppression
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        
        # Draw bounding boxes and labels
        detected_objects = []
        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                confidence = confidences[i]
                color = self.colors[class_ids[i]]
                
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                detected_objects.append({
                    'label': label,
                    'confidence': confidence,
                    'bbox': [x, y, w, h],
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
        
        return frame, detected_objects
    
    def stream_from_esp32(self):
        """Stream video from ESP32-CAM and perform object detection"""
        try:
            # Connect to ESP32-CAM stream
            response = requests.get(self.stream_url, stream=True)
            bytes_data = bytes()
            
            for chunk in response.iter_content(chunk_size=1024):
                bytes_data += chunk
                a = bytes_data.find(b'\xff\xd8')  # JPEG start
                b = bytes_data.find(b'\xff\xd9')  # JPEG end
                
                if a != -1 and b != -1:
                    jpg = bytes_data[a:b+2]
                    bytes_data = bytes_data[b+2:]
                    
                    # Decode JPEG
                    frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    
                    if frame is not None:
                        # Perform object detection
                        detected_frame, detections = self.detect_objects(frame)
                        
                        # Update latest frame and results
                        self.latest_frame = detected_frame
                        self.detection_results = detections
                        
        except Exception as e:
            print(f"Error streaming from ESP32-CAM: {e}")
    
    def generate_frames(self):
        """Generate frames for web streaming"""
        while True:
            if self.latest_frame is not None:
                ret, buffer = cv2.imencode('.jpg', self.latest_frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.1)

# Initialize object detector
detector = ObjectDetector()

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(detector.generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detections')
def get_detections():
    """Get latest detection results"""
    return {'detections': detector.detection_results}

if __name__ == '__main__':
    # Start ESP32-CAM streaming in a separate thread
    stream_thread = threading.Thread(target=detector.stream_from_esp32)
    stream_thread.daemon = True
    stream_thread.start()
    
    # Start Flask web server
    app.run(host='0.0.0.0', port=5000, debug=True)

