from typing import Literal
import mediapipe as mp
from enum import Enum
import time
import cv2

# Enum
class Mode(Enum):
    HAND = "Hand"
    FACE = "Face"
    Cup = "Cup"
    Chair = "Chair"
    Shoe = "Shoe"
    Camera = "Camera"
    Laptop = "Laptop"
    Bottle = "Bottle"
                 

class Tracker:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils

        # Start video capture
        self.camera = cv2.VideoCapture(0)
            

    def hand_detection(self, coordinates: bool = False):

        # Initializations
        mp_hands = mp.solutions.hands
        coordinates_dict = {}

        prev_time = 0


        with mp_hands.Hands(
            min_detection_confidence=0.8,
            min_tracking_confidence=0.5,
        ) as self.hands:
            
            while self.camera.isOpened():
                ret, frame = self.camera.read()
                if not ret:
                    break

                # Convert the BGR image to RGB
                self.image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.image.flags.writeable = False

                # Make detection
                self.results = self.hands.process(self.image)

                # Convert the image back to BGR
                self.image.flags.writeable = True
                self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)

                if self.results.multi_hand_landmarks:
                    for self.hand_landmarks in self.results.multi_hand_landmarks:
                        self.mp_drawing.draw_landmarks(
                            self.image, self.hand_landmarks, mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                            self.mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2)
                        )

                        if coordinates:
                            for id, landmark in enumerate(self.hand_landmarks.landmark):
                                height, width, _ = self.image.shape
                                x, y = int(landmark.x * width), int(landmark.y * height)
                                coordinates_dict[f"landmark_{id}"] = (x, y)
                                cv2.putText(self.image, f"{id}", (x, y),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)
                                # Display output in console
                                print(f"Landmark {id}: ({x}, {y})")
                
                # FPS Display
                curr_time = time.time()
                fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
                prev_time = curr_time
                cv2.putText(self.image, f'FPS: {int(fps)}', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow('Hand Tracking', self.image)

                yield coordinates_dict if coordinates else None

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

        # return coordinates_dict if coordinates else None
    

    def face_detection(self, coordinates: bool = False):
        # Initialize MediaPipe Face Mesh and drawing utilities
        mp_face_mesh = mp.solutions.face_mesh

        # Drawing specifications for the landmarks
        drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

        coordinates_dict = {}

        prev_time = 0

        # Initialize FaceMesh model
        with mp_face_mesh.FaceMesh(
            max_num_faces=2,                   # Detect up to 2 faces (you can increase)
            refine_landmarks=True,             # Includes iris landmarks (eyes)
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as face_mesh:

            while self.camera.isOpened():
                success, self.image = self.camera.read()
                if not success:
                    break

                # Convert BGR to RGB
                self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
                self.image.flags.writeable = False

                # Process the image (detect face landmarks)
                self.results = face_mesh.process(self.image)

                # Convert RGB back to BGR for display
                self.image.flags.writeable = True
                self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)

                # Draw face mesh landmarks
                if self.results.multi_face_landmarks:
                    for self.face_landmarks in self.results.multi_face_landmarks:
                        self.mp_drawing.draw_landmarks(
                            image=self.image,
                            landmark_list=self.face_landmarks,
                            connections=mp_face_mesh.FACEMESH_TESSELATION,  # full face mesh
                            landmark_drawing_spec=None,
                            connection_drawing_spec=drawing_spec)
                        
                        if coordinates:
                            h, w, _ = self.image.shape
                            for id, landmark in enumerate(self.face_landmarks.landmark):
                                x, y = int(landmark.x * w), int(landmark.y * h)
                                coordinates_dict[f"landmark_{id}"] = (x, y)
                        
                # FPS Display
                curr_time = time.time()
                fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
                prev_time = curr_time
                cv2.putText(self.image, f'FPS: {int(fps)}', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)   

                # Display output
                cv2.imshow('Face Mesh', self.image)

                yield coordinates_dict if coordinates else None

                # Exit on pressing 'q'
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break

        # return coordinates_dict if coordinates else None

    def object_detection(self, model_name: Literal["Cup", "Chair", "Shoe", "Camera", "Laptop", "Bottle"] = "Cup"):
        
        # Initializations for object detection can be added here
        mp_objectron = mp.solutions.objectron

        with mp_objectron.Objectron(
            static_image_mode=False,
            max_num_objects=5,
            min_detection_confidence=0.5,
            model_name=model_name  # Example model, can be changed
        ) as objectron:

            while self.camera.isOpened():
                ret, frame = self.camera.read()
                if not ret:
                    break

                # Convert the BGR image to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                # Make detection
                results = objectron.process(image)

                # Convert the image back to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Draw landmarks
                if results.detected_objects:
                    for detected_object in results.detected_objects:
                        self.mp_drawing.draw_landmarks(
                            image, detected_object.landmarks_2d, mp_objectron.BOX_CONNECTIONS)
                        self.mp_drawing.draw_axis(image, detected_object.rotation, detected_object.translation)

                cv2.imshow('Object Detection', image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break


    def release(self):
        self.camera.release()
        cv2.destroyAllWindows()

    def run(self, mode: Literal["Hand", "Face", "Cup", "Chair", "Shoe", "Camera", "Laptop", "Bottle"] = "Hand", coordinates: bool = False):
        try: 
            mode_enum = Mode(mode) # converts "1" -> Mode.HAND except ValueError: raise ValueError("Invalid mode! Choose '1' for HAND or '2' for FACE.")

            if mode_enum == Mode.HAND:
                return self.hand_detection(coordinates=coordinates)
            elif mode_enum == Mode.FACE:
                return self.face_detection(coordinates=coordinates)
            elif mode_enum in [Mode.Cup, Mode.Chair, Mode.Shoe, Mode.Camera, Mode.Laptop, Mode.Bottle]:
                return self.object_detection(model_name=f"{mode_enum.value}")

        except ValueError:
            raise ValueError("Invalid mode!")
        
        finally:
            self.release()


# if __name__ == "__main__":
#     tracker = Tracker()
#     try:

#         # Change mode here to Mode.FACE for face detection
#         # coords = tracker.run(mode="Hand", coordinates=True)
#         for coords in tracker.face_detection(coordinates=True):

#             print("Coordinates:", coords)

#     finally:
#         tracker.release()
