
import serial
import time
import cv2

# Connect to Arduino
arduinoData = serial.Serial('/dev/cu.usbmodem11101', 9600)
time.sleep(2)  # Allow Arduino to reset after serial connection

# Start camera
capture = cv2.VideoCapture(0)

if not capture.isOpened():
    print("Error: Could not open camera.")
    arduinoData.close()
    raise SystemExit

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

if face_cascade.empty():
    print("Error: Could not load face detection model.")
    capture.release()
    arduinoData.close()
    raise SystemExit

try:
    while True:
        isTrue, frame = capture.read()

        if not isTrue:
            print("Could not read camera frame.")
            break

        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray, 1.05, 8, minSize=(120, 120)
        )

        for (x, y, w, h) in faces:
            # Find the center of the detected face
            center_x = x + w // 2
            center_y = y + h // 2

            cv2.rectangle(
                frame, (x, y), (x + w, y + h), (0, 255, 0), 2
            )
            cv2.circle(
                frame, (center_x, center_y), 5, (0, 0, 255), -1
            )

            # Scale camera coordinates to 1920 x 1080
            arduino_x = int(center_x * 1920 / width)
            arduino_y = int(center_y * 1080 / height)

            arduinoData.write(f"{arduino_x},{arduino_y}\r".encode())
            print(f"X: {arduino_x}, Y: {arduino_y}")

            break  # Track the first detected face

        cv2.imshow('Face Tracking', frame)

        if cv2.waitKey(20) & 0xFF == ord('d'):
            break

finally:
    capture.release()
    cv2.destroyAllWindows()
    arduinoData.close()
