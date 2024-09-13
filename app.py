import PIL.ImageGrab
import time, mediapipe, cv2
import HandTracker as ht
import math
import pyautogui
import PIL
import uuid


def takeScreenshot():
    unique_id = uuid.uuid4()
    screenshot = pyautogui.screenshot()
    screenshot.save(f"/home/talha/Pictures/Screenshot_{unique_id}.png")
    time.sleep(2)


pTime = 0
cTime = 0
cap = cv2.VideoCapture(0)
detector = ht.HandDetector(detectionCon=0.7,trackingCon=0.7)
while True:
    success, img = cap.read()

    img = detector.findHands(img)
    position = detector.findPos(img)

    if len(position) != 0:
        iid, icx, icy = position[8]
        tid, tcx, tcy = position[4]

        # Findding the mid point of both X, Y axis
        mcx, mcy = (icx + tcx)//2, (icy + tcy)//2
        length = math.hypot((icx - tcx), (icy - tcy))
        print(length)
        cv2.circle(img,(icx,icy),15,color=(255,0,255),thickness=cv2.FILLED)
        cv2.circle(img,(tcx,tcy),15,color=(255,0,255),thickness=cv2.FILLED)
        cv2.circle(img,(mcx, mcy),15,color=(255,0,255),thickness=cv2.FILLED)
        if length < 17:
            takeScreenshot()

        cv2.line(img, (icx, icy), (tcx, tcy), color=(255,255,0), thickness=5)




    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
    # Show the image with landmarks drawn
    cv2.imshow("Image", img)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
