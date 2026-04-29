import torch
import numpy as np
import time
import sys
import cv2 
import os

# --- 1. IMPORT YOUR CUSTOM MODULES ---
try:
    from model import FatigueLSTM, EnsembleFusion
    from advisor import InjuryAdvisor
    from database import get_athlete_historical_risk  # Integrated Database Layer
except ImportError as e:
    print(f"Error: {e}")
    sys.exit()

# --- 2. INITIALIZATION ---
print("[DEMO MODE] Bypassing MediaPipe due to Python 3.14 incompatibility...")
advisor = InjuryAdvisor()
lstm_net = FatigueLSTM()
model = EnsembleFusion(lstm_net)

# Load weights safely from Desktop
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
model_path = os.path.join(desktop_path, "ensemble_fusion.pth")

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    print(f"[SYSTEM] Neural Weights loaded successfully.")
else:
    print("[WARNING] Weights not found, using random initialization.")

model.eval()

# Buffer for 30-frame window
frame_buffer = []
last_advice_time = 0 
display_frame = np.zeros((480, 640, 3), dtype=np.uint8)

print("\n>>> SYSTEM ACTIVE: PULLING FROM POSTGRES & RUNNING ENSEMBLE <<<")

try:
    while True:
        # 1. SIMULATE LANDMARKS 
        # Mimicking the 33 landmarks (x,y,z) MediaPipe would provide
        noise = np.random.normal(0.5, 0.05, 99) 
        frame_buffer.append(noise.tolist())

        if len(frame_buffer) == 30:
            input_tensor = torch.tensor([frame_buffer], dtype=torch.float32)
            
            # 2. DATABASE INTEGRATION
            # Pulling the 0.595 historical factor you just verified
            historical_risk = get_athlete_historical_risk("user_01")
            
            with torch.no_grad():
                # ENSEMBLE FUSION: LSTM (Live) + Database (Historical)
                risk_score_tensor = model(input_tensor, tabular_risk_score=historical_risk)
                
                # Visual scaling for the demo
                risk_percent = min(risk_score_tensor.item() * 100 * 1.5, 100.0)
            
            # --- UI FEEDBACK ---
            display_frame[:] = (0, 0, 0) 
            color = (0, 255, 0) # Green
            status = "OPTIMAL"
            
            if risk_percent > 55: 
                color = (0, 255, 255) # Yellow
                status = "FATIGUE DETECTED"
            if risk_percent > 75: 
                color = (0, 0, 255) # Red
                status = "HIGH INJURY RISK"

            # Draw Pro Dashboard
            cv2.rectangle(display_frame, (0,0), (640, 80), (30, 30, 30), -1)
            cv2.putText(display_frame, "SYNAPSE HACKATHON: FOUR ACES", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            cv2.putText(display_frame, f"LIVE RISK: {risk_percent:.1f}%", (50, 180), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.8, color, 4)
            cv2.putText(display_frame, f"STATUS: {status}", (50, 250), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.putText(display_frame, f"DB HISTORICAL FACTOR: {historical_risk}", (50, 350), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            # --- GEMINI AI TRIGGER ---
            if risk_percent > 75 and (time.time() - last_advice_time > 20):
                print(f"\n[AI ALERT] Risk at {risk_percent:.1f}% - Querying Gemini Advisor...")
                advice = advisor.get_advice(
                    round(risk_percent, 1), 
                    "Knee Valgus detected in spatial simulation.", 
                    "Temporal decay in stride frequency via LSTM."
                )
                print(f"GEMINI ADVICE: {advice}\n")
                last_advice_time = time.time()
            
            frame_buffer.pop(0) 

        cv2.imshow('Four Aces Demo', display_frame)
        
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass

cv2.destroyAllWindows()
