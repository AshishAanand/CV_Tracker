---

## 🧠 Real-Time Multi-Mode Tracker (Hands, Face, Objects)

### 🎯 Overview

The **Tracker** is a powerful, modular, and extensible computer vision class built using **Mediapipe** and **OpenCV**.
It allows you to **track hands, faces, and 3D objects in real time**, returning landmark coordinates for each mode with a clean, reusable design.

This project serves as the **foundation for gesture recognition, virtual control systems, and AI-based computer vision** applications.

---

### ⚡ Features

* 🖐️ **Hand Tracking** – Detects and tracks 21 hand landmarks in real time.
* 😀 **Face Detection** – Tracks facial landmarks and bounding boxes.
* 📦 **3D Object Tracking** – Detects and estimates 3D object positions (e.g., cup, camera, chair).
* 🧩 **Modular Design** – Each mode (hand, face, object) is easily extendable.
* 🧠 **Coordinate Access** – Returns and prints real-time landmark coordinates.
* 🎥 **Camera Integration** – Works seamlessly with your webcam or external video input.

---

### 🧱 Class Architecture

```python
import mediapipe as mp
import cv2
from enum import Enum

class Mode(Enum):
    HAND = "hand"
    FACE = "face"
    OBJECT = "object"

class Tracker:
    def __init__(self, mode: Mode):
        self.mode = mode
        # Initialize Mediapipe modules here
        # (Hands, Face Detection, or Objectron)
    
    def track(self, frame):
        """
        Process the given frame and return tracking results.
        """
        pass

    def get_coordinates(self, results):
        """
        Extract and return landmark coordinates.
        """
        pass
```

---

### 🚀 Usage Example

```python
import cv2
from tracker import Tracker, Mode

# Initialize the tracker
tracker = Tracker(Mode.HAND)

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    # Get processed frame and coordinates
    results, coords = tracker.track(frame)

    # Display the results
    cv2.imshow("Tracker", frame)
    print("Final Coordinates:", coords)  # prints list of landmark coordinates

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
```

---

### 🧩 Supported Modes

| Mode          | Description                                  | Mediapipe Module         |
| ------------- | -------------------------------------------- | ------------------------ |
| `Mode.HAND`   | Tracks 21 hand landmarks                     | `mp.solutions.hands`     |
| `Mode.FACE`   | Tracks facial landmarks                      | `mp.solutions.face_mesh` |
| `Mode.OBJECT` | Tracks 3D objects like cups, cameras, chairs | `mp.solutions.objectron` |

---

### 📦 Requirements

Install dependencies using:

```bash
pip install mediapipe opencv-python
```

---

### 🧠 Potential Use Cases

* 🤖 Gesture-Controlled Systems (Virtual Mouse, Volume Control)
* 🕹️ Vision-Based Games
* 🧤 Sign Language Recognition
* 🎨 Virtual Drawing Tools
* 👁️ AI Jarvis Interaction (Gesture-based activation)
* 🦾 Robotics Control via Camera Input

---

### 🧭 Project Structure

```
CVTracker/
│
├── tracker.py         # Main Tracker class
├── controller(Hand).py            # Example usage / test script
└── README.md          # You’re reading it :)
```

---

### 💡 Future Enhancements

* Add **gesture classification** using ML models
* Add **voice integration** (Jarvis trigger via gesture + speech)
* Integrate **AR overlays** for visual feedback

---

### 🧑‍💻 Author

**Ashish Anand**
💬 *AI Developer • Game Creator • Future Robotics Engineer*
🔗 [GitHub](https://github.com/AshishAanand) | [LinkedIn](https://www.linkedin.com/in/ashish-anand-49b958311/)

---
