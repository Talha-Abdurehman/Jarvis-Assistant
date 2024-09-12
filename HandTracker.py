import cv2
import mediapipe as mp
import time


class HandDetector():
    
    def __init__(self, mode=False, maxHands=2,complexity=1, detectionCon=0.5, trackingCon=0.5, ):
        self.mode = mode
        self.maxHands = maxHands
        self.complexity = complexity
        self.detectionCon = detectionCon
        self.trackingCon = trackingCon


        self.mpHands = mp.solutions.hands
        self.mpDraw = mp.solutions.drawing_utils
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.complexity, self.detectionCon, self.trackingCon)
    
    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        # If hand landmarks are detected
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:

                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img
                    #         cv2.circle(img, (cx,cy), 15, (255,0,255), cv2.FILLED)
                    #         # pyautogui.moveTo(cx, cy)
                    #         # print(f"Current Mouse Loc: {pyautogui.position()}")
                    #         cv2.line(img, (cx, cy),(0,0), color=(255,0,255), thickness=2)
    def findPos(self, img, hand_no=0, draw=True):
        
        lmlist = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[hand_no]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmlist.append([id, cx, cy])
                if draw:
                    cv2.circle(img,(cx,cy), 5, (255,0.255), cv2.FILLED)
        return lmlist


def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    while True:
        success, img = cap.read()

        img = detector.findHands(img)
        position = detector.findPos(img)

        if len(position) != 0:
            print(position[2])

        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
        # Show the image with landmarks drawn
        cv2.imshow("Image", img)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    
if __name__ == "__main__":
    main()