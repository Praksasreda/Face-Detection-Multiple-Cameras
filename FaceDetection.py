import cv2
import insightface
from insightface.app import FaceAnalysis

app = FaceAnalysis(providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0,det_size=(640,640))

cap = cv2.VideoCapture(0)

while True:
    ret,frame = cap.read()
    faces = app.get(frame)

    for face in faces:
        bbox = face.bbox.astype(int)
        cv2.rectangle(frame,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,255,0),2)


    cv2.imshow("Detection",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()