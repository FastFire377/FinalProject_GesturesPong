import cvzone
import time


class FPS:
    """
    FPS class for calculating and displaying the Frames Per Second in a video stream.

    Attributes:
        pTime (float): Previous time stamp.
        frameTimes (list): List to keep track of frame times.
        avgCount (int): Number of frames over which to average the FPS.
    """

    def __init__(self, avgCount=30):
        """
        Initialize FPS class.

        :param avgCount: Number of frames over which to average the FPS, default is 30.
        """
        self.pTime = time.time()  # Initialize previous time to current time
        self.frameTimes = []  # List to store the time taken for each frame
        self.avgCount = avgCount  # Number of frames to average over

    def update(self, img=None, pos=(20, 50), bgColor=(255, 0, 255),
               textColor=(255, 255, 255), scale=3, thickness=3):
        """
        Update the frame rate and optionally display it on the image.

        :param img: Image to display FPS on. If None, just returns the FPS value.
        :param pos: Position to display FPS on the image.
        :param bgColor: Background color of the FPS text.
        :param textColor: Text color of the FPS display.
        :param scale: Font scale of the FPS text.
        :param thickness: Thickness of the FPS text.
        :return: FPS value, and optionally the image with FPS drawn on it.
        """

        cTime = time.time()  # Get the current time
        frameTime = cTime - self.pTime  # Calculate the time difference between the current and previous frame
        self.frameTimes.append(frameTime)  # Append the time difference to the list
        self.pTime = cTime  # Update previous time

        # Remove the oldest frame time if the list grows beyond avgCount
        if len(self.frameTimes) > self.avgCount:
            self.frameTimes.pop(0)

        avgFrameTime = sum(self.frameTimes) / len(self.frameTimes)  # Calculate the average frame time
        fps = 1 / avgFrameTime  # Calculate FPS based on the average frame time

        # Draw FPS on image if img is provided
        if img is not None:
            cvzone.putTextRect(img, f'FPS: {int(fps)}', pos,
                               scale=scale, thickness=thickness, colorT=textColor,
                               colorR=bgColor, offset=10)
        return fps, img