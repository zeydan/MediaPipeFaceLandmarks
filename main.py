import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

mpFace = mp.solutions.face_mesh
faces = mpFace.FaceMesh()
mpDraw = mp.solutions.drawing_utils

while True:
    _, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faces.process(imgRGB)

    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            for id, lm in enumerate(faceLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)

                if id == 10: # top
                    cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)
                if id == 152: # bottom
                    cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)
                if id == 234: # right
                    cv2.circle(img, (cx, cy), 10, (0, 0, 255), cv2.FILLED)
                if id == 454: # left
                    cv2.circle(img, (cx, cy), 10, (0, 255, 255), cv2.FILLED)
        
            mpDraw.draw_landmarks(img, faceLms, mpFace.FACEMESH_TESSELATION)

    img = cv2.flip(img, 1) # to see image like mirror
    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()