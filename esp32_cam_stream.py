# ESP32-CAM Video Streaming Code
# This code should be uploaded to the ESP32-CAM module

import network
import socket
import camera
import time

# Wi-Fi credentials
SSID = "your_wifi_ssid"
PASSWORD = "your_wifi_password"

# Initialize camera
camera.init(0, format=camera.JPEG, framesize=camera.FRAME_VGA)

def connect_wifi():
    """Connect to Wi-Fi network"""
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    wlan.connect(SSID, PASSWORD)
    
    while not wlan.isconnected():
        print("Connecting to Wi-Fi...")
        time.sleep(1)
    
    print("Connected to Wi-Fi")
    print("IP Address:", wlan.ifconfig()[0])
    return wlan.ifconfig()[0]

def start_stream_server():
    """Start HTTP server for video streaming"""
    ip = connect_wifi()
    
    # Create socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((ip, 80))
    s.listen(5)
    
    print(f"Server started at http://{ip}")
    
    while True:
        conn, addr = s.accept()
        print(f"Connection from {addr}")
        
        request = conn.recv(1024)
        
        # HTTP response headers
        response = "HTTP/1.1 200 OK\r\n"
        response += "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n"
        response += "\r\n"
        
        conn.send(response.encode())
        
        try:
            while True:
                # Capture frame
                buf = camera.capture()
                
                # Send frame
                conn.send(b"--frame\r\n")
                conn.send(b"Content-Type: image/jpeg\r\n")
                conn.send(f"Content-Length: {len(buf)}\r\n\r\n".encode())
                conn.send(buf)
                conn.send(b"\r\n")
                
                time.sleep(0.1)  # 10 FPS
                
        except Exception as e:
            print(f"Error: {e}")
        finally:
            conn.close()

if __name__ == "__main__":
    start_stream_server()

