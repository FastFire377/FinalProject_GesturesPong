import glob
import os
import cv2
import mediapipe as mp
from mediapipe.tasks import python
import threading
import time 

class GestureRecognizer:
    def __init__(self, images):
        self.images
        self.lock = threading.Lock()
        self.current_gestures = []
        self.selected_option = None
        self.start_time = None
        self.required_duration = 2  # manter gesto por 2 segundos
        self.options = {
            "Thumb_Up": "Jogar",
            "Victory": "Pontuações",
            "Thumb_Down": "Créditos"
        }
    
    def main(self):
        num_hands = 2
        model_path = "C:/Users/admin/OneDrive - Universidade do Algarve/EngenhariaSistemasTecnologiasInformaticas/ComputacaoVisual/ProjetoFinal/repoClone/FinalProject_GesturesPong/model/gesture_recognizer.task"
        GestureRecognizer = mp.tasks.vision.GestureRecognizer
        GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        
        """
        0 - Unrecognized gesture, label: Unknown
        1 - Closed fist, label: Closed_Fist
        2 - Open palm, label: Open_Palm
        3 - Pointing up, label: Pointing_Up
        4 - Thumbs down, label: Thumb_Down
        5 - Thumbs up, label: Thumb_Up
        6 - Victory, label: Victory
        7 - Love, label: ILoveYou
        """
        
        options = GestureRecognizerOptions(
            base_options=python.BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.LIVE_STREAM,
            num_hands = num_hands,
            result_callback=self.__result_callback)
        recognizer = GestureRecognizer.create_from_options(options)

        timestamp = 0 
        mp_drawing = mp.solutions.drawing_utils
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=num_hands,
                min_detection_confidence=0.65,
                min_tracking_confidence=0.65)

        cap = cv2.VideoCapture(0)

        while cv2.pollKey() == -1: # cv2.waitKey(1) & 0xFF == 27
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            np_array = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np_array)
                    recognizer.recognize_async(mp_image, timestamp)
                    timestamp = timestamp + 1 # should be monotonically increasing, because in LIVE_STREAM mode
                    
                self.put_gestures(frame)
                
            self.display_menu(frame)
            cv2.imshow('MediaPipe Hands', frame)
            

        cap.release()


    def display_menu(self, frame):
        # Display menu options
        y_pos = 50
        for gesture, option in self.options.items():
            cv2.putText(frame, f"{gesture}: {option}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (255, 255, 255), 2, cv2.LINE_AA)
            y_pos += 50
        frame[100:150 + 10, 100:100 + 10] = images
        # Display the selected option
        if self.selected_option:
            cv2.putText(frame, f"Selected: {self.selected_option}", (10, y_pos + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    
    def put_gestures(self, frame):
        self.lock.acquire()
        gestures = self.current_gestures
        self.lock.release()
        y_pos = 50
        for hand_gesture_name in gestures:
            # show the prediction on the frame
            cv2.putText(frame, hand_gesture_name, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                                1, (0,0,255), 2, cv2.LINE_AA)
            y_pos += 50

    def __result_callback(self, result, output_image, timestamp_ms):
        #print(f'gesture recognition result: {result}')
        self.lock.acquire() # solves potential concurrency issues
        self.current_gestures = []
        if result and result.gestures:
            # Get the most confident gesture
            gesture_name = result.gestures[0][0].category_name

            if gesture_name in self.options:
                current_time = time.time()

                # Check if the gesture is consistent
                if self.start_time and self.current_gestures and self.current_gestures[0] == gesture_name:
                    duration = current_time - self.start_time
                    if duration >= self.required_duration:
                        self.selected_option = self.options[gesture_name]
                        print(f"Selected: {self.selected_option}")
                        self.start_time = None  # Reset the timer
                        self.current_gestures = []
                else:
                    # Start the timer for a new gesture
                    self.start_time = current_time
                    self.current_gestures = [gesture_name]
            else:
                self.start_time = None  # Reset if unrecognized gesture
                self.current_gestures = []
        """
        if result is not None and any(result.gestures):
            print("Recognized gestures:")
            for single_hand_gesture_data in result.gestures:
                gesture_name = single_hand_gesture_data[0].category_name
                print(gesture_name)
                self.current_gestures.append(gesture_name)
        """
        self.lock.release()

if __name__ == "__main__":
    images = {}
    images_path = "images/"
    for images_file in glob.glob(os.path.join(images_path, "*.png")):
        images_name = os.path.splitext(os.path.basename(images_file))[0]
        print("image name:", images_name)
        images[images_name] = cv2.imread(images_file, cv2.IMREAD_UNCHANGED)
    
    rec = GestureRecognizer(images=images)
    rec.main()