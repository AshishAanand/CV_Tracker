from tracker import Tracker
import cv2
import mediapipe as mp


if __name__ == "__main__":
    tracker = Tracker()
    try:
        # coords = tracker.run(mode="Hand", coordinates=True)
        for coords in tracker.face_detection(coordinates=True):

            print("Coordinates:", coords)

    finally:
        tracker.release()
