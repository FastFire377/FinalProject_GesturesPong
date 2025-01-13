import glob
import os
import cv2
import cvzone
import mediapipe as mp
from mediapipe.tasks import python
import threading
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import time

class GestureRecognizer:
    def __init__(self, images):
        self.images = images
        self.lock = threading.Lock()
        self.current_gestures = []
        self.selected_option = None
        self.start_time = None
        self.required_duration = 1  # manter gesto por 1 segundo
        self.menu = True #Display menu boolean
        self.inGame = False
        self.menuOptions = {
            "Thumb_Up": "Jogar",
            "Victory": "Pontuacao",
            "Thumb_Down": "Creditos"
        }
        
        """
        Possible Gestures by mediapipe:
        0 - Unrecognized gesture, label: Unknown
        1 - Closed fist, label: Closed_Fist
        2 - Open palm, label: Open_Palm
        3 - Pointing up, label: Pointing_Up
        4 - Thumbs down, label: Thumb_Down
        5 - Thumbs up, label: Thumb_Up
        6 - Victory, label: Victory
        7 - Love, label: ILoveYou
        """
        
        num_hands = 2
        model_path = "FinalProject_GesturesPong/model/gesture_recognizer.task"
        GestureRecognizer = mp.tasks.vision.GestureRecognizer
        GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        
        self.options = GestureRecognizerOptions(
            base_options=python.BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.LIVE_STREAM,
            num_hands = num_hands,
            result_callback=self.__result_callback)
        self.recognizer = GestureRecognizer.create_from_options(self.options)

        self.timestamp = 0 
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=num_hands,
                min_detection_confidence=0.65,
                min_tracking_confidence=0.65)
        
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 1280)
        self.cap.set(4, 720)
        
    def main(self):
        
        while cv2.pollKey() == -1: # cv2.waitKey(1) & 0xFF == 27
            ret, frame = self.cap.read()
            frame = cv2.flip(frame, 1)
            if not ret:
                break
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            np_array = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np_array)
                    self.recognizer.recognize_async(mp_image, self.timestamp)
                    self.timestamp = self.timestamp + 1
                    
                self.put_gestures(frame)
            if self.menu:
                self.display_menu(frame)
            cv2.imshow('Pong Project', frame)
            
        self.cap.release()
        cv2.destroyAllWindows()


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


    def display_menu(self, frame):
        # Display menu options
        y_pos = 50
        
        frame = cvzone.overlayPNG(frame, self.images["JogarButton"], [310, -50])
        frame = cvzone.overlayPNG(frame, self.images["PontuacaoButton"], [280, 173])
        frame = cvzone.overlayPNG(frame, self.images["CreditosButton"], [280, 420])
        
        # Display the selected option
        if self.selected_option:
            cv2.putText(frame, f"Selected: {self.selected_option}", (10, y_pos + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            self.current_gestures = []
            self.menu = False
            
            if self.selected_option == "Jogar":
                self.inGame = True
                self.Jogar()
                self.inGame = False
            elif self.selected_option == "Creditos":
                self.Creditos()
                
            elif self.selected_option == "Pontuacao":
                self.Pontuacao()
                
            
            # Reinicializar variáveis
            self.menu = True
            self.selected_option = None


    def Creditos(self):
        pass
    
    
    def Pontuacao(self):
        pass


    def __result_callback(self, result, output_image, timestamp_ms):
        self.lock.acquire()  

        if result and result.gestures:
            gesture_name = result.gestures[0][0].category_name
            print("gesture name:", gesture_name)
            current_time = time.time()
            if gesture_name in self.menuOptions:
                
                # print("current_time:", current_time)
                # Check if the gesture is consistent
                if self.start_time and self.current_gestures and self.current_gestures[0] == gesture_name:
                    duration = current_time - self.start_time
                    if duration >= self.required_duration and not self.inGame:
                        self.selected_option = self.menuOptions[gesture_name]
                        print(f"Selected: {self.selected_option}")
                        self.start_time = None  # Reset the timer
                        self.current_gestures = []
                else:
                    self.start_time = current_time 
                    self.current_gestures = [gesture_name]
            else:
                # Start the timer for a new gesture
                self.start_time = None
                self.current_gestures = []
        self.lock.release()

        
    def Jogar(self):
        # Importing all images
        imgBackground = cv2.imread("FinalProject_GesturesPong/images/backgroundPong.png")
        imgGameOver = cv2.imread("FinalProject_GesturesPong/images/GameOver.png")
        imgBall = cv2.imread("FinalProject_GesturesPong/images/bolaresize.png", cv2.IMREAD_UNCHANGED)
        imgBlock1 = cv2.imread("FinalProject_GesturesPong/images/Bloco1.png", cv2.IMREAD_UNCHANGED)
        imgBlock2 = cv2.imread("FinalProject_GesturesPong/images/Bloco2.png", cv2.IMREAD_UNCHANGED)
        
        self.current_gestures = []

        ballPos = [100, 100]
        speedX = 15
        speedY = 15
        gameOver = False
        score = [0, 0]
        
        while cv2.pollKey() == -1:
            ret, frame = self.cap.read()
            frame = cv2.flip(frame, 1)
            if not ret:
                break
            
            results = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            np_array = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np_array)
                    self.recognizer.recognize_async(mp_image, self.timestamp)
                    self.timestamp = self.timestamp + 1
                self.put_gestures(frame)
            
            #Guardar frame antes de adicionar background
            rawFrame = frame.copy()
            
            # Overlaying the background image
            frame = cv2.addWeighted(rawFrame, 0.2, imgBackground, 0.8, 0)

            # Check for hands
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    x_min = min([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1]
                    y_min = min([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0]
                    y_min = np.clip(int(y_min - 50), 20, 415)

                    if x_min < frame.shape[1] // 2:  # Left hand
                        frame = cvzone.overlayPNG(frame, imgBlock1, (59, y_min))
                        if 59 < ballPos[0] < 59 + imgBlock1.shape[1] and y_min < ballPos[1] < y_min + imgBlock1.shape[0]:
                            speedX = -speedX
                            ballPos[0] += 30
                            score[0] += 1
                            speedX += 0.2
                            #speedY += 0.2

                    else:  # Right hand
                        frame = cvzone.overlayPNG(frame, imgBlock2, (1195, y_min))
                        if 1195 - 50 < ballPos[0] < 1195 and y_min < ballPos[1] < y_min + imgBlock2.shape[0]:
                            speedX = -speedX
                            ballPos[0] -= 30
                            score[1] += 1
                            
            if ballPos[0] < 40 or ballPos[0] > 1260:
                gameOver = True
                self.current_gestures = []
                self.update_highest_score(score[1] + score[0])
                ballPos = [100, 100]
                speedX, speedY = 15, 15
                score = [0, 0]

            if not gameOver:
                # Change ball position frame by frame 
                if ballPos[1] >= 500 or ballPos[1] <= 10:
                    speedY = -speedY

                ballPos[0] += speedX
                ballPos[1] += speedY
                
                ballPos = [int(ballPos[0]), int(ballPos[1])]

                # Draw the ball
                frame = cvzone.overlayPNG(frame, imgBall, ballPos)

                cv2.putText(frame, str(score[0]), (300, 650), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 5)
                cv2.putText(frame, str(score[1]), (900, 650), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 5)

            # game over 
            else:
                frame = cv2.addWeighted(rawFrame, 0.7, imgGameOver, 0.3, 0)
                cv2.putText(frame, str(score[1] + score[0]).zfill(2), (585, 360), cv2.FONT_HERSHEY_SIMPLEX,
                            2.5, (200, 0, 200), 5)
                
                if self.current_gestures: 
                    gesture_name = self.current_gestures[0]
                    if gesture_name == "Thumb_Up":
                        self.current_gestures = []
                        print("Restarting game... with callback")
                        gameOver = False
                    elif gesture_name == "Thumb_Down":
                        print("Exiting Game")
                        break

            frame[580:700, 20:233] = cv2.resize(rawFrame, (213, 120))

            cv2.imshow('Pong Project', frame)
        
        

    def update_highest_score(self, score):
        # Nome do ficheiro para guardar o maior score
        scorefilepath = "FinalProject_GesturesPong/highestScore.txt"

        # Verificar o maior score atual no ficheiro
        if os.path.exists(scorefilepath):
            with open(scorefilepath, 'r') as file:
                try:
                    highest_score = int(file.read().strip())
                except ValueError:
                    highest_score = 0  # If no value
        else:
            highest_score = 0

        # Atualizar o ficheiro se o score atual for maior
        if score > highest_score:
            with open(scorefilepath, 'w') as file:
                file.write(str(score))
            print("Highest score updated:", score)
            return True
        else:
            return False


if __name__ == "__main__":
    images = {}
    images_path = "FinalProject_GesturesPong/images/"
    for images_file in glob.glob(os.path.join(images_path, "*.png")):
        images_name = os.path.splitext(os.path.basename(images_file))[0]
        # print("image name:", images_name)
        images[images_name] = cv2.imread(images_file, cv2.IMREAD_UNCHANGED)
    
    rec = GestureRecognizer(images=images)
    rec.main()