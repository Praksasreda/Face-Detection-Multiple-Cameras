import cv2
import threading
import insightface
from insightface.app import FaceAnalysis
import numpy as np
import sqlite3
import pickle
import time
from elevenlabs.client import ElevenLabs
from elevenlabs import stream
from dotenv import load_dotenv
import os 

load_dotenv()
# ── CONFIG ──────────────────────────────────────────────
API_KEY = os.getenv("ELEVENLABS_API_KEY")
VOICE_ID =  os.getenv("ELEVENLABS_VOICE_ID")
CAMERA_INDEXES = [0] # se dodaja glede na stevilo kamer
RECOGNITION_THRESHOLD = 0.5
COOLDOWN_SECOND = 10000
DB_PATH = "isac.db"
# ────────────────────────────────────────────────────────

# InsightFace
app = FaceAnalysis(providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# ElevenLabs
eleven = ElevenLabs(api_key=API_KEY)

# ── DATABASE ─────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS persons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            role TEXT NOT NULL,
            embedding BLOB NOT NULL,
            registered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_name TEXT,
            camera_index INTEGER,
            detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def save_person(name, role, embedding):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO persons (name, role, embedding) VALUES (?, ?, ?)",
              (name, role, pickle.dumps(embedding)))
    conn.commit()
    conn.close()

def load_persons():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT name, role, embedding FROM persons")
    rows = c.fetchall()
    conn.close()
    return [{"name": r[0], "role": r[1], "embedding": pickle.loads(r[2])} for r in rows]

def log_detection(name, camera_index):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO detections (person_name, camera_index) VALUES (?, ?)",
              (name, camera_index))
    conn.commit()
    conn.close()

# ── TTS ──────────────────────────────────────────────────
tts_lock = threading.Lock()

def speak(text):
    def _speak():
        with tts_lock:
            audio = eleven.text_to_speech.convert(
                voice_id=VOICE_ID,
                text=text,
                model_id="eleven_monolingual_v1"
            )
            stream(audio)
    threading.Thread(target=_speak, daemon=True).start()

def isac_phrase(name, role, known=True):
    if not known:
        return "Unidentified individual."
    if role == "Driver":
        return f"{name}, welcome back."
    return f"Identifying passenger. {name}."

# ── RECOGNITION ──────────────────────────────────────────
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def recognize(embedding, db):
    best_match = None
    best_score = -1
    for person in db:
        score = cosine_similarity(embedding, person["embedding"])
        if score > best_score:
            best_score = score
            best_match = person
    if best_score >= RECOGNITION_THRESHOLD:
        return best_match
    return None

# ── REGISTRACIJA ─────────────────────────────────────────
def register_face(name, role="Passenger"):
    cap = cv2.VideoCapture(0)
    print(f"Registering '{name}' as {role}. Press SPACE to capture (multiple times for better accuracy), Q to finish.")
    embeddings = []

    while True:
        ret, frame = cap.read()
        cv2.putText(frame, f"Registering: {name} | Captures: {len(embeddings)} | SPACE=capture Q=done",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("Register", frame)

        key = cv2.waitKey(1)
        if key == ord(' '):
            faces = app.get(frame)
            if len(faces) == 0:
                print("No face detected, try again.")
                continue
            embeddings.append(faces[0].embedding)
            print(f"Captured {len(embeddings)} embedding(s).")

        elif key == ord('q'):
            if len(embeddings) == 0:
                print("No captures, aborting.")
            else:
                avg_embedding = np.mean(embeddings, axis=0)
                save_person(name, role, avg_embedding)
                print(f"Registered '{name}' ({role}) with {len(embeddings)} captures.")
            break

    cap.release()
    cv2.destroyAllWindows()

# ── KAMERA THREAD ─────────────────────────────────────────
class Camera:
    def __init__(self, index):
        self.index = index
        self.frame = None
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            print(f"Warning: Camera {index} not available.")
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while True:
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    self.frame = frame

    def release(self):
        self.cap.release()

# ── MAIN LOOP ─────────────────────────────────────────────
def run():
    init_db()
    db = load_persons()

    if len(db) == 0:
        print("Baza je prazna. Najprej registriraj vsaj eno osebo.")
        name = input("Ime: ")
        role = input("Role (Driver/Passenger): ")
        register_face(name, role)
        db = load_persons()

    cameras = [Camera(i) for i in CAMERA_INDEXES]
    last_spoken = {}

    print("Press Q to quit, R to register new face.")

    while True:
        for cam in cameras:
            if cam.frame is None:
                continue

            frame = cam.frame.copy()
            faces = app.get(frame)

            for face in faces:
                bbox = face.bbox.astype(int)
                match = recognize(face.embedding, db)

                if match:
                    name = match["name"]
                    role = match["role"]
                    label = f"{name} | {role}"
                    color = (0, 255, 0)

                    now = time.time()
                    key = f"{name}"
                    if key not in last_spoken or now - last_spoken[key] > COOLDOWN_SECOND:
                        last_spoken[key] = now
                        speak(isac_phrase(name, role))
                        log_detection(name, cam.index)
                else:
                    label = "UNKNOWN"
                    color = (0, 0, 255)
                    key = f"{name}"
                    now = time.time()
                    if key not in last_spoken or now - last_spoken[key] > COOLDOWN_SECOND:
                        last_spoken[key] = now
                        speak(isac_phrase(None, None, known=False))
                        log_detection("unknown", cam.index)

                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                cv2.putText(frame, label, (bbox[0], bbox[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.imshow(f"ISAC CAM {cam.index}", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            name = input("Ime: ")
            role = input("Role (Driver/Passenger): ")
            register_face(name, role)
            db = load_persons()  # reload

    for cam in cameras:
        cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run()