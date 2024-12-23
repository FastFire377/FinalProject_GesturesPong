import cv2 as cv
from mediapipe.python.solutions.holistic import Holistic
import mediapipe


class MediaPipe:
    def __init__(self, video_source=0):
        self.cap = cv.VideoCapture(video_source)
        
        self.mp_holistic = mediapipe.solutions.holistic
        self.holistic_model = self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initializing the drawing utils for drawing the facial landmarks on image
        self.mp_drawing = mediapipe.solutions.drawing_utils


    def draw_fps(self, t_start, frame):
        """Calcula e desenha FPS"""
        t_end = cv.getTickCount()
        fps = cv.getTickFrequency() / (t_end - t_start)

        cv.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 25),
            cv.FONT_HERSHEY_SIMPLEX,
            fontScale=0.8,
            color=(0, 128, 26),
            thickness=1,
            lineType=cv.LINE_AA,
        )


    def process_frame(self, frame):
        results = self.holistic_model.process(frame)
        
        self.mp_drawing.draw_landmarks(
            frame,
            results.face_landmarks,
            self.mp_holistic.FACEMESH_CONTOURS,
            self.mp_drawing.DrawingSpec(
                color=(255,0,255),
                thickness=1,
                circle_radius=1
            ),
            self.mp_drawing.DrawingSpec(
                color=(0,255,255),
                thickness=1,
                circle_radius=1
            )
        )
    
        # Drawing Right hand Land Marks
        self.mp_drawing.draw_landmarks(
        frame, 
        results.right_hand_landmarks, 
        self.mp_holistic.HAND_CONNECTIONS
        )
    
        # Drawing Left hand Land Marks
        self.mp_drawing.draw_landmarks(
        frame, 
        results.left_hand_landmarks, 
        self.mp_holistic.HAND_CONNECTIONS
        )
        
        return frame



    def run(self):
        try:
            while cv.pollKey() == -1:
                t_start = cv.getTickCount()

                success, frame = self.cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    continue

                frame = self.process_frame(frame)
                self.draw_fps(t_start, frame)
                cv.imshow("MediaPipe", frame)

        finally:
            self.cap.release()
            cv.destroyAllWindows()


# Executa o programa
if __name__ == "__main__":

    mediapipe = MediaPipe(video_source=0)
    mediapipe.run()
