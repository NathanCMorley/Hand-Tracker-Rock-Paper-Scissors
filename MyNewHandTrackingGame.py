import cv2
import time
import HandTrackingModule as htm
import numpy as np
import pyautogui
import mediapipe as mp
import os
import random


def pickGesture(rocks,papers,sizors):
    rockPercent = rocks/(rocks+papers+sizors)*100 
    paperPercent = papers/(rocks+papers+sizors)*100
    sizorsPercent = sizors/(rocks+papers+sizors)*100
    paperBoundry = rockPercent + paperPercent
    randomPick = random.Random().randint(0, 100)
    print(paperBoundry)

    if(randomPick <= rockPercent):
        return 2
    elif(randomPick < paperBoundry):
        return 3
    else:
        return 1

def gestureToNum(gesture):
    if(gesture == "rock"):
        return 1
    elif(gesture == "paper"):
        return 2
    elif(gesture == "scissors"):
        return 3

def numToGesture(num):
    if(num == 1):
        return "Rock"
    elif(num == 2):
        return "Paper"
    elif(num == 3):
        return "Scissors"

def drawText(img, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=2, thickness=3, color=(255, 0, 0), shadow_color=(0, 0, 0)):
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = position[0] - text_size[0] // 2
    text_y = position[1] + text_size[1] // 2

    # Draw shadow
    cv2.putText(img, text, (text_x + 2, text_y + 2), font, font_scale, shadow_color, thickness + 2, cv2.LINE_AA)
    # Draw text
    cv2.putText(img, text, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)

def decideWinner(gesture, computerGesture, img,streak):
    h, w, _ = img.shape
    center_x, center_y = w // 2, h // 2
    
    if gesture == computerGesture:
        text = "Tie"
    elif gesture == 1 and computerGesture == 2:
        text = "You Lose"
        
        streak = 0
    elif gesture == 1 and computerGesture == 3:
        text = "You Win"
        
        streak += 1
    elif gesture == 2 and computerGesture == 1:
        text = "You Win"
        
        streak += 1
    elif gesture == 2 and computerGesture == 3:
        text = "You Lose"
      
        streak = 0
    elif gesture == 3 and computerGesture == 1:
        text = "You Lose"
        
        streak = 0
    elif gesture == 3 and computerGesture == 2:
        text = "You Win"
        
        streak += 1
    else:
        text = "Error"

    text+= " Computer picked " + numToGesture(computerGesture)
    drawText(img, text, (center_x, center_y + 40),color=(0xFF, 0xFF, 0xFF), shadow_color=(0, 0, 0))

    

   
    return streak

def main():
    f = open("record.txt", "r")
    values = f.read().split(" ")
    f.close()
    rocks = int(values[0])
    papers = int(values[1])
    sizors = int(values[2])
    allTimeHighestStreak = int(values[3])
    cap = cv2.VideoCapture(0)
    bufferPick = 0
    currentPick = 0
    pTime = 0
    cTime = 0
    detector = htm.handDetector()
    isFrontFacing = True
    pickUID = random.Random().randint(0, 1000000000000000)
    directory = "/Users/nathanmorley/documents/projects/HandtrackerAi/rps_data_sample/thumbsUp"
    RPS = 1
    playerGesture = 999999
    waitingForGesture = False
    success, img = cap.read()
    h, w, _ = img.shape
    center_x, center_y = w // 2, h // 2
    streak = 0

    # Initialize MediaPipe Hands and Gesture Recognizer
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
    
    # Update the path to the gesture recognizer model
    model_path = os.path.join(os.path.dirname(__file__), 'gesture_recognizer.task')
    mp_gesture = mp.tasks.vision.GestureRecognizer.create_from_options(
        mp.tasks.vision.GestureRecognizerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path)
        )
    )

    while True:

        success, img = cap.read()
        if isFrontFacing:
            img = cv2.flip(img, 1)
        
        # Convert the BGR image to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        drawText(img, "Streak: " + str(streak), (150, 50), color=(0xFF, 0xFF, 0xFF), shadow_color=(0, 0, 0))
        if playerGesture != 999999:
            decideWinner(playerGesture, computerGesture, img,0)
        if waitingForGesture:
            drawText(img, "Shoot!", (center_x, center_y - 40),color=(0xFF, 0xFF, 0xFF), shadow_color=(0, 0, 0))
            
        results = hands.process(img_rgb)
        
        pickedGesture = "None"

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                mp.solutions.drawing_utils.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                # Extract landmark positions
                lmlist = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                fingersUp = detector.findFingerUp(lmlist)
                # Recognize gestures
                gesture_results = mp_gesture.recognize(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
                )
                # print('Gesture results:')
                for gesture_list in gesture_results.gestures:
                    for gesture in gesture_list:
                        print(f"Gesture recognized: {gesture.category_name}")
                        pickedGesture = gesture.category_name

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        currentPick = time.time()
        if(currentPick - bufferPick > 0.75):
            RPS+=1
            bufferPick = currentPick

        

        if RPS == 1:
            text = "Rock"
        elif RPS == 2:
            text = "Paper"
        elif RPS == 3:
            text = "Scissors"
        elif RPS == 4:
            text = "Shoot!"
            waitingForGesture = True
        else:
            text = ""

        if text:
            drawText(img, text, (center_x, center_y - 40),color=(0xFF, 0xFF, 0xFF), shadow_color=(0, 0, 0))

        
        if((pickedGesture != "None") & (RPS >= 4) & (pickedGesture != "none") & (pickedGesture != "thumbsUp") & (pickedGesture != "")):
            waitingForGesture = False
            print ("Picked Gesture")
            print(pickedGesture)
            
            # print(gestureToNum(pickedGesture))
            playerGesture = gestureToNum(pickedGesture)
            print (gesture)
            computerGesture = pickGesture(rocks,papers,sizors)
            streak = decideWinner(playerGesture, computerGesture,img,streak)
            if(streak > allTimeHighestStreak):
                allTimeHighestStreak = streak
            print(streak)
            print("Computer Gesture: " + numToGesture(computerGesture))
            print("Gesture: " + numToGesture(playerGesture))
        

            

            if playerGesture == 1:
                rocks += 1
            elif playerGesture == 2:
                papers += 1
            elif playerGesture == 3:
                sizors += 1
            RPS = 0
            f = open("record.txt", "w")
            f.write(str(rocks) + " " + str(papers) + " " + str(sizors) + " " + str(allTimeHighestStreak))


        
        
        


        #     cv2.imwrite(f"thumbsUp{pickUID}.png", img)
        #     pickUID +=1
        #     print("Pick")

      

        
        cv2.imshow("Image", img)
        cv2.waitKey(1)
        
    

if __name__ == "__main__":
    main()

