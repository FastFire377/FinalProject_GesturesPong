import cv2 as cv
from mediapipe.python.solutions.holistic import Holistic


class MediaPipe:
    def __init__(self, video_source=0):
        self.holistic = Holistic()
        self.cap = cv.VideoCapture(video_source)


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
        results = self.holistic.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))

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
