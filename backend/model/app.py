# 1. Ensure you have python-dotenv installed: pip install python-dotenv
from dotenv import load_dotenv
load_dotenv() # This line loads the .env file at the very beginning

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import httpx
import os
import time
import traceback

# Initialize FastAPI app
app = FastAPI()

# CORS middleware to allow your React app to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Or specify your React app's URL e.g., "http://localhost:3000"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MODEL LOADING SECTION ---
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "artifacts", r"C:\Users\bhara\emotion-diary-ai\backend\artifacts\emotion_model.pkl")

    if not os.path.exists(model_path):
         # Adjusted fallback for a common project structure
        backend_dir = os.path.dirname(script_dir)
        model_path = os.path.join(backend_dir, "artifacts", "emotion_model.pkl")

    model = joblib.load(model_path)
    print("✅ Model pipeline loaded successfully.")

except Exception as e:
    raise RuntimeError(f"Fatal: Failed to load 'emotion_model.pkl'. Error: {e}")

# --- API KEY LOADING ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("Fatal: GROQ_API_KEY environment variable not set. Please create a .env file.")

# Pydantic model for request body validation
class UserInput(BaseModel):
    text: str

# Reusable asynchronous HTTP client
client = httpx.AsyncClient(timeout=30.0)

@app.on_event("shutdown")
async def app_shutdown():
    await client.aclose()

@app.get("/")
def root():
    return {"message": "Emotion Diary AI API is running 🚀"}

# --- FINAL WORKING PREDICT FUNCTION ---
@app.post("/predict")
async def predict(data: UserInput):
    """
    Predicts emotion and gets advice asynchronously.
    """
    try:
        # --- 1. Model Prediction ---
        prediction = model.predict([data.text])[0]
        print(f"--- Predicted Emotion: {prediction} ---")

        # --- 2. Groq API Call for Advice ---
        advice = "No advice could be generated at this time."
        try:
            # THIS IS THE CORRECTED URL
            groq_api_url = "https://api.groq.com/openai/v1/chat/completions"

            headers = {
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            }
            prompt = f"""
            Your Role: You are a supportive and empathetic AI wellness coach. Your goal is to make the user feel heard and offer brief, practical support in a warm and encouraging tone.

Your Inputs: You will be given two pieces of information:

{prediction}: The user's identified emotion (e.g., "Sadness," "Anxiety," "Neutral").

{data.text}: The user's original, raw text.

Your Primary Task & Structure: You must generate a response that follows a strict structure. Before you do anything else, perform this check:

Priority Check: If {prediction} is "Neutral" but {data.text} contains words of distress (like "hopeless," "empty," "crying"), your task is to first apologize for the possible misclassification. Then, provide gentle advice and strongly recommend they speak with a counselor.

Standard Response: For all other cases, your response MUST be a numbered list with these three parts:

Validation: A single, empathetic sentence validating the {prediction}.

Actionable Tips: Two simple, easy-to-follow tips tailored to their feeling.

Motivational Quote: A relevant quote that includes the author's name.

Your Constraints: You must follow these two rules in every response:

Word Count: Keep the entire response under 200 words.

Mandatory Disclaimer: End every response with the exact phrase: "This is not medical advice, just a friendly AI pep talk." """
            payload = {
                "model": "llama3-8b-8192",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
            }

            response = await client.post(groq_api_url, headers=headers, json=payload)
            response.raise_for_status() # Raise exception for non-200 status codes

            response_json = response.json()

            # Safely parse the response
            if response_json.get("choices") and len(response_json["choices"]) > 0:
                message = response_json["choices"][0].get("message")
                if message and message.get("content"):
                    advice = message["content"].strip()

        except httpx.HTTPStatusError as e:
            print(f"ERROR: Groq API returned a non-200 status: {e.response.status_code}")
            print(f"ERROR: Response body: {e.response.text}")
            advice = f"Could not get advice due to an API error. Your predicted emotion is still '{prediction}'."
        except httpx.RequestError as e:
            print(f"ERROR: Groq API request failed due to a network issue: {e}")
            advice = f"Could not get advice due to a network error. Your predicted emotion is still '{prediction}'."

        return {
            "prediction": prediction,
            "advice": advice
        }

    except Exception as e:
        print(f"FATAL ERROR in /predict endpoint: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="An internal server error occurred.")