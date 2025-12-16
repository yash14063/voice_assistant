import cv2
import mediapipe as mp
import os
import time
import threading
import queue
import win32com.client as wincl # Standard Windows library

# Disable oneDNN warning
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ---------------- ROBUST VOICE ENGINE (WINDOWS SAPI5) ----------------
# We use direct Windows Dispatch instead of pyttsx3 to avoid event loop freezing
tts_queue = queue.Queue()

def tts_worker():
    """
    Worker thread that directly accesses Windows Voice API.
    This bypasses the 'run loop' issues of pyttsx3 entirely.
    """
    # Initialize the Windows Speaker
    try:
        speaker = wincl.Dispatch("SAPI.SpVoice")
    except Exception as e:
        print(f"Voice Error: {e}")
        return

    while True:
        text = tts_queue.get()
        if text is None: break
        
        try:
            # This is a blocking call, but since we are in a thread,
            # it won't freeze the camera!
            speaker.Speak(text)
        except Exception as e:
            print(f"TTS Error: {e}")
            
        tts_queue.task_done()

# Start the TTS thread
threading.Thread(target=tts_worker, daemon=True).start()

def speak(text):
    # Clear queue to ensure real-time response (remove old pending alerts)
    with tts_queue.mutex:
        tts_queue.queue.clear()
    tts_queue.put(text)

# ---------------- MEDIAPIPE ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

TIP_IDS = [4, 8, 12, 16, 20]

# ---------------- FINGER COUNT FUNCTION ----------------
def count_fingers(hand_landmarks, results):
    fingers_up = 0
    lm = hand_landmarks.landmark

    if results.multi_handedness:
        handedness = results.multi_handedness[0].classification[0].label
    else:
        return 0

    # Thumb Logic
    if handedness == "Right": 
        if lm[4].x < lm[3].x:
            fingers_up += 1
    else:
        if lm[4].x > lm[3].x:
            fingers_up += 1

    # Other four fingers
    for i in range(1, 5):
        if lm[TIP_IDS[i]].y < lm[TIP_IDS[i] - 2].y:
            fingers_up += 1

    return fingers_up

# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(0)

last_spoken = ""
fist_start_time = None
SOS_TIME = 2  # seconds

last_speech_time = 0
SPEECH_COOLDOWN = 3.0 

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    action_text = "Scanning..."
    finger_count = -1

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        finger_count = count_fingers(hand_landmarks, results)
        current_time = time.time()

        # ---------------- LOGIC MAPPING ----------------
        if finger_count == 1:
            action_text = "YES"
            phrase = "Yes"
        elif finger_count == 2:
            action_text = "NO"
            phrase = "No"
        elif finger_count == 3:
            action_text = "Request WATER"
            phrase = "I need water"
        elif finger_count == 4:
            action_text = "HELP"
            phrase = "I need help please assist me"
        elif finger_count == 5:
            action_text = "STOP"
            phrase = "Please stop"
        elif finger_count == 0:
            action_text = "Holding..."
            phrase = None 
        
        # ---------------- SPEECH TRIGGER ----------------
        if finger_count in [1, 2, 3, 4, 5]:
            fist_start_time = None
            if action_text != last_spoken or (current_time - last_speech_time > SPEECH_COOLDOWN):
                speak(phrase)
                last_spoken = action_text
                last_speech_time = current_time

        # ---------------- EMERGENCY SOS ----------------
        if finger_count == 0:  
            if fist_start_time is None:
                fist_start_time = time.time()
            
            elapsed = time.time() - fist_start_time
            countdown = int(SOS_TIME - elapsed) + 1
            
            cv2.putText(image, f"SOS in: {countdown}", (400, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            if elapsed >= SOS_TIME:
                action_text = "🚨 EMERGENCY SOS 🚨"
                if int(elapsed * 10) % 2 == 0: 
                    image[:] = (0, 0, 255)
                
                if current_time - last_speech_time > 2.0:
                    speak("Emergency! Please help!")
                    last_speech_time = current_time
        else:
            fist_start_time = None

    # ---------------- DISPLAY ----------------
    cv2.rectangle(image, (0,0), (640, 80), (0,0,0), -1)
    
    cv2.putText(image, f"Fingers: {finger_count if finger_count != -1 else '-'}",
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    color = (0, 255, 0) if finger_count != 0 else (0, 0, 255)
    cv2.putText(image, action_text,
                (250, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Vision Based Smart Assistant", image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()