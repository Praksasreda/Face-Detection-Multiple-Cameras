# ISAC Passenger Identification System

> *"Identifying passenger. Admin detected."*

Real-time facial recognition system for vehicles, styled after the ISAC AI from The Division 2. Identifies the driver and passengers via camera feed and responds with voice confirmation using a cloned ISAC voice through ElevenLabs.

Built as a portfolio project exploring edge AI, computer vision, and voice synthesis.

---

## Demo

- Detects faces across multiple cameras simultaneously
- Matches faces against a local SQLite database
- Speaks ISAC-style confirmations via ElevenLabs TTS
- Logs all detections with timestamps

---

## How It Works

```
Camera(s) → OpenCV → InsightFace (face detection + embedding) → SQLite (recognition) → ElevenLabs (voice)
```

On first run, the system prompts you to register at least one face. Each registered person gets an averaged embedding from multiple captures for better accuracy. Detection runs across all connected cameras in parallel threads, with a global cooldown per person to prevent duplicate announcements.

---

## Requirements

- Python 3.9+
- Webcam or USB camera
- ElevenLabs account (requires paid subscription)

---

## Setup

**1. Clone the repository**
```bash
git clone https://github.com/praksasreda/isac-passenger-id
cd isac-passenger-id
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Configure environment variables**

Create a `.env` file in the project root:
```
ELEVENLABS_API_KEY=your_api_key
ELEVENLABS_VOICE_ID=your_voice_id
```

To get your Voice ID: ElevenLabs dashboard → Voices → select voice → copy ID.

**4. Run**
```bash
python RecognitionMultipleCamsv2.py
```

On first run, the system will guide you through face registration before starting detection.

---

## Controls

| Key | Action |
|-----|--------|
| `SPACE` | Capture frame during registration |
| `R` | Register a new face while detection is running |
| `Q` | Quit |

---

## Configuration

At the top of `main.py`:

```python
CAMERA_INDEXES = [0, 1]       # 0 = built-in webcam, 1 = USB camera
RECOGNITION_THRESHOLD = 0.5   # lower = stricter matching
COOLDOWN_SECONDS = 10         # seconds between repeat announcements
```

To identify which camera index corresponds to which physical camera, run:
```bash
python -c "
import cv2
for i in range(4):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f'Camera {i} available')
    cap.release()
"
```

---

## Stack

| Component | Library |
|-----------|---------|
| Camera capture | OpenCV |
| Face detection & recognition | InsightFace |
| ONNX inference | onnxruntime |
| Voice synthesis | ElevenLabs |
| Local database | SQLite3 |
| Environment config | python-dotenv |

---

## Roadmap


- [ ] ESP32-CAM support for wireless cameras
- [ ] Raspberry Pi 5 deployment (in-vehicle edge computing)
- [ ] HUD overlay
- [ ] Consent flow for new face registration

---

## Notes

- Face embeddings are stored locally in `isac.db` — never leaves your machine
- ElevenLabs API key is required for voice output; without it the system will not start
- InsightFace downloads models (~500MB) on first run

---

## License

MIT
