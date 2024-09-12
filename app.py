import cv2
import mediapipe as mp
import threading
import time
import mouse




# Initialize video capture and mediapipe components
cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
mpDraw = mp.solutions.drawing_utils
hands = mpHands.Hands()

# Function to process the frames
def process_frame():

    pTime = 0
    cTime = 0
    while True:
        success, img = cap.read()
        if not success:
            break
        
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        # If hand landmarks are detected
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):
                    # print(id, lm)
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    # print(id,cx, cy)

                    if id == 8:
                        print(cx, cy)
                        mouse.move(cx, cy)
                        print(f"Current Mouse Loc: {mouse.get_position()}")
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
        # Show the image with landmarks drawn
        cv2.imshow("Image", img)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Run the process in a separate thread
thread = threading.Thread(target=process_frame)
thread.start()

# Join the thread to ensure it closes cleanly
thread.join()

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

