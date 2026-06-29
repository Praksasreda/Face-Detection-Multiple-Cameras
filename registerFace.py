import cv2
import insightface
from insightface.app import FaceAnalysis
import numpy as np
import pickle

app = FaceAnalysis(providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0,det_size=(640,640))

def register_face(name,role="Passanger"):
    cap = cv2.VideoCapture(0)
    print(f"Registiring {name}. Press space to capture")

    while True:
        ret,frame = cap.read()
        cv2.imshow("Register",frame)

        key = cv2.waitKey(1)
        if key == ord(' '):
            faces = app.get(frame)
            if len(faces) == 0:
                print("No face detected, try again")
                continue


            embeddings = []
            while len(embeddings)<10:
                ret,frame = cap.read()
                faces = app.get(frame)
                if len(faces)> 0:
                    embeddings.append(faces[0].embedding)

            # nalozi novo/obstojeco bazo

            avg_embedding = np.mean(embeddings,axis=0)

            try:
                with open("faces.pkl","rb") as f:
                    db = pickle.load(f)
            
            except FileNotFoundError:
                db = []

            db.append({"frame":name,"role":role,"embedding": avg_embedding})

            with open("faces.pkl","wb") as f:
                pickle.dump(db,f)

            print(f"Registered: {name} ({role})")
            break

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()





