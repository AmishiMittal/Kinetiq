import os
import time
from google import genai
from google.genai import errors

# --- CONFIGURATION ---
API_KEY = os.getenv(
    "GEMINI_API_KEY",
    "AIzaSyAbiy5Pp_h75E1mEpLH_lORsfY47WZFxqc",  # legacy fallback; prefer setting GEMINI_API_KEY
)

class InjuryAdvisor:
    def __init__(self):
        # Initialize the 2026 GenAI Stack
        if not API_KEY:
            raise RuntimeError("GEMINI_API_KEY is not configured.")

        self.client = genai.Client(api_key=API_KEY)
        self.primary_model = "gemini-2.0-flash"
        self.backup_model = "gemini-1.5-flash"

    def get_advice(self, risk_score, spatial_notes, temporal_notes):
        """
        Takes ensemble data and generates professional injury prevention advice.
        """
        prompt = f"""
        CONTEXT: Real-time pose estimation and injury risk prediction.
        DATA:
        - Risk Score: {risk_score}%
        - Spatial Feature Branch: {spatial_notes}
        - Temporal Feature Branch: {temporal_notes}
        
        TASK:
        Provide a concise 2-sentence recovery/corrective protocol for the athlete.
        Keep it professional and action-oriented.
        """

        try:
            # Attempt 1: Gemini 2.0 (High Performance)
            response = self.client.models.generate_content(
                model=self.primary_model,
                contents=prompt
            )
            return response.text

        except errors.ClientError as e:
            # Attempt 2: Fallback to 1.5 if 429 (Rate Limit) occurs
            if "429" in str(e):
                print("\n[SYSTEM LOG] 2.0 Quota exceeded. Using 1.5 Backup...")
                try:
                    response = self.client.models.generate_content(
                        model=self.backup_model,
                        contents=prompt
                    )
                    return response.text
                except Exception as backup_err:
                    return f"Backup Error: {backup_err}"
            else:
                return f"API Error: {e}"

# --- RUN THE TEST ---
if __name__ == "__main__":
    # This simulates how your main script will call the advisor
    try:
        advisor = InjuryAdvisor()
    except Exception as e:
        print(f"Unable to initialize Gemini advisor: {e}")
        raise SystemExit(1)
    
    # Simulated Ensemble Fusion Results
    risk = 74.8
    spatial = "Medial knee collapse (valgus) detected during landing phase."
    temporal = "Fatigue markers: Increased ground contact time and asymmetric stride."

    print("Generating real-time advice...")
    print("-" * 30)
    print(advisor.get_advice(risk, spatial, temporal))
    print("-" * 30)
