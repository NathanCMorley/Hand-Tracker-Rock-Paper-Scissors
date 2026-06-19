import cv2
import mediapipe as mp
import time

# cap = cv2.VideoCapture(0)
class handDetector():
    def __init__(self, mode=False, maxHands=2, minDetectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.minDetectionCon = minDetectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode, 
                                        max_num_hands=self.maxHands, 
                                        min_detection_confidence=self.minDetectionCon, 
                                        min_tracking_confidence=self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def isRightHand(self, lmList):
        if len(lmList) != 0:
            # Check the relative position of the thumb and pinky
            # Thumb tip is landmark 4, pinky tip is landmark 20
            if lmList[4][1] < lmList[20][1]:
                return True
        return False

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        return lmList
    
    def findFingerUp(self, lmList):
        results = [False, False, False, False, False]
        if(len(lmList) != 0):
           
            results[0] = self.isThumbUp(lmList)
            
            results[1] = self.isIndexFingerUp(lmList)
           
            results[2] = self.isMiddleFingerUp(lmList)
            
            results[3] = self.isRingFingerUp(lmList)
           
            results[4] = self.isPinkyUp(lmList)
           
        return results

    def isThumbUp(self, lmList):
        if len(lmList) != 0:
            # Thumb tip is landmark 4, thumb IP joint is landmark 3
            return not(lmList[4][1] > lmList[3][1])
        return True

    def isIndexFingerUp(self, lmList):
        if len(lmList) != 0:
            # Index finger tip is landmark 8, PIP joint is landmark 6
            return lmList[8][2] < lmList[6][2]
        return False

    def isMiddleFingerUp(self, lmList):
        if len(lmList) != 0:
            # Middle finger tip is landmark 12, PIP joint is landmark 10
            return lmList[12][2] < lmList[10][2]
        return False

    def isRingFingerUp(self, lmList):
        if len(lmList) != 0:
            # Ring finger tip is landmark 16, PIP joint is landmark 14
            return lmList[16][2] < lmList[14][2]
        return False

    def isPinkyUp(self, lmList):
        if len(lmList) != 0:
            # Pinky finger tip is landmark 20, PIP joint is landmark 18
            return lmList[20][2] < lmList[18][2]
        return False

def drawText(img, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=2, thickness=3, color=(255, 0, 255), shadow_color=(0, 0, 0)):
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = position[0] - text_size[0] // 2
    text_y = position[1] + text_size[1] // 2

    # Draw shadow
    cv2.putText(img, text, (text_x + 2, text_y + 2), font, font_scale, shadow_color, thickness + 2, cv2.LINE_AA)
    # Draw text
    cv2.putText(img, text, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)

def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    cTime = 0
    detector = handDetector()

    while True:
        success, img = cap.read()
        img = detector.findHands(img, draw=True)
        lmlist = detector.findPosition(img)
        if len(lmlist) != 0:
            if detector.isThumbUp(lmlist):
                print("Thumb is up")
            if detector.isIndexFingerUp(lmlist):
                print("Index finger is up")
            if detector.isMiddleFingerUp(lmlist):
                print("Middle finger is up")
            if detector.isRingFingerUp(lmlist):
                print("Ring finger is up")
            if detector.isPinkyUp(lmlist):
                print("Pinky is up")
            if detector.isRightHand(lmlist):
                print("Right hand detected")
            else:
                print("Left hand detected")

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        h, w, _ = img.shape

        drawText(img, str(int(fps)), (w - 100, 50))

        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()