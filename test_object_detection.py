import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

class ObjectDetectionTester:
    def __init__(self):
        # For testing purposes, we'll use a simplified approach
        # In a real implementation, you would load actual YOLO weights
        self.classes = [
            "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
            "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
            "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
            "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
            "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
            "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
            "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
            "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
            "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
            "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
            "toothbrush"
        ]
        
        # Colors for different classes
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
        
    def simulate_detection(self, image):
        """Simulate object detection for demonstration purposes"""
        height, width = image.shape[:2]
        
        # Simulate some detections
        detections = []
        
        # Add some random detections for demonstration
        if np.random.random() > 0.3:  # 70% chance of detection
            num_detections = np.random.randint(1, 4)
            
            for _ in range(num_detections):
                # Random class
                class_id = np.random.randint(0, len(self.classes))
                label = self.classes[class_id]
                confidence = np.random.uniform(0.6, 0.95)
                
                # Random bounding box
                x = np.random.randint(0, width // 2)
                y = np.random.randint(0, height // 2)
                w = np.random.randint(50, width // 3)
                h = np.random.randint(50, height // 3)
                
                # Ensure box is within image bounds
                x = min(x, width - w)
                y = min(y, height - h)
                
                detections.append({
                    'label': label,
                    'confidence': confidence,
                    'bbox': [x, y, w, h],
                    'class_id': class_id
                })
        
        return detections
    
    def draw_detections(self, image, detections):
        """Draw bounding boxes and labels on the image"""
        result_image = image.copy()
        
        for detection in detections:
            x, y, w, h = detection['bbox']
            label = detection['label']
            confidence = detection['confidence']
            class_id = detection['class_id']
            color = self.colors[class_id]
            
            # Draw bounding box
            cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 2)
            
            # Draw label
            label_text = f"{label} {confidence:.2f}"
            label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Draw label background
            cv2.rectangle(result_image, (x, y - label_size[1] - 10), 
                         (x + label_size[0], y), color, -1)
            
            # Draw label text
            cv2.putText(result_image, label_text, (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return result_image
    
    def test_with_sample_images(self):
        """Test object detection with sample images"""
        # Create sample images for testing
        test_images = []
        
        # Create a simple test image
        img1 = np.ones((480, 640, 3), dtype=np.uint8) * 128
        cv2.rectangle(img1, (100, 100), (300, 300), (0, 255, 0), -1)  # Green rectangle
        cv2.circle(img1, (500, 200), 50, (255, 0, 0), -1)  # Blue circle
        test_images.append(("Sample Image 1", img1))
        
        # Create another test image
        img2 = np.ones((480, 640, 3), dtype=np.uint8) * 200
        cv2.rectangle(img2, (50, 50), (200, 200), (0, 0, 255), -1)  # Red rectangle
        cv2.rectangle(img2, (400, 300), (600, 450), (255, 255, 0), -1)  # Yellow rectangle
        test_images.append(("Sample Image 2", img2))
        
        results = []
        
        for name, image in test_images:
            # Simulate object detection
            detections = self.simulate_detection(image)
            
            # Draw detections
            result_image = self.draw_detections(image, detections)
            
            results.append({
                'name': name,
                'original': image,
                'result': result_image,
                'detections': detections
            })
        
        return results
    
    def save_results(self, results):
        """Save test results as images"""
        os.makedirs('test_results', exist_ok=True)
        
        for i, result in enumerate(results):
            # Save original image
            cv2.imwrite(f'test_results/original_{i+1}.jpg', result['original'])
            
            # Save result image
            cv2.imwrite(f'test_results/detected_{i+1}.jpg', result['result'])
            
            print(f"Saved results for {result['name']}")
            print(f"Detections: {len(result['detections'])}")
            for detection in result['detections']:
                print(f"  - {detection['label']}: {detection['confidence']:.2f}")
            print()

def main():
    """Main testing function"""
    print("VisionEdge Object Detection Test")
    print("=" * 40)
    
    tester = ObjectDetectionTester()
    
    # Run tests
    results = tester.test_with_sample_images()
    
    # Save results
    tester.save_results(results)
    
    print("Test completed successfully!")
    print("Check the 'test_results' directory for output images.")

if __name__ == "__main__":
    main()

