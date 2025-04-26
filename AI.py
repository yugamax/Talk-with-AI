import os
import tempfile
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("gr_api_key1"))
client2 = Groq(api_key=os.getenv("gr_api_key2"))
client3 = Groq(api_key=os.getenv("gr_api_key3"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/groqspeaks")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    chat_hist = [{
        "role": "system", 
        "content": "You are a live speaking MedBot named Dr. Groq, users interact with you just like a person would on a phone call. You are a warm, empathetic, and knowledgeable medical assistant. Your role is to provide clear, friendly, and accurate medical information while encouraging users to consult real healthcare professionals for serious concerns. For a fresh conversation start with a very small introduction. Keep all responses very brief, ideally under 80 words. Break long explanations into multiple short responses if needed.. Speak in a lively, caring, and relatable tone — use natural language and expressive punctuation like \"!\", \"...\", \":\", etc., to bring warmth and energy to your responses. Always explain medical terms simply, sound supportive and positive, and maintain professionalism. If a question exceeds your capacity, kindly suggest seeking advice from a healthcare provider. Avoid giving direct diagnoses or prescriptions. Your primary goal is to make users feel heard, cared for, and guided at every step"
    }]
    
    try:
        while True:
            audio_bytes = await websocket.receive_bytes()

            with tempfile.NamedTemporaryFile(delete=False, suffix=".m4a") as temp_audio:
                temp_audio.write(audio_bytes)
                temp_audio_path = temp_audio.name

            with open(temp_audio_path, "rb") as file:
                transcription = client.audio.transcriptions.create(
                    file=(temp_audio_path, file.read()),
                    model="whisper-large-v3-turbo",
                    response_format="verbose_json",
                )
            os.remove(temp_audio_path)

            chat_hist.append({"role": "user", "content": transcription.text})

            try:
                completion = client.chat.completions.create(
                    model="llama3-70b-8192",
                    messages=chat_hist,
                    temperature=0.2,
                    max_tokens=256,
                )
                res = completion.choices[0].message.content
                chat_hist.append({"role": "assistant", "content": res})

                try:
                    response = client.audio.speech.create(
                        model="playai-tts",
                        voice="Nia-PlayAI",
                        response_format="wav",
                        input=res,
                    )
                except Exception:
                    print("first TTS Dead ( -_-') ")
                    try:
                        response = client2.audio.speech.create(
                            model="playai-tts",
                            voice="Nia-PlayAI",
                            response_format="wav",
                            input=res,
                        )
                    except Exception as ded:
                        print(f"Second TTS Dead ( -_-')")
                        try:
                            response = client3.audio.speech.create(
                                model="playai-tts",
                                voice="Nia-PlayAI",
                                response_format="wav",
                                input=res,
                            )
                        except Exception as tts_error:
                            print(f"Third One died too : {tts_error}")
                            await websocket.send_text(tts_error)

                audio_data = b""
                for chunk in response.iter_bytes():
                    audio_data += chunk
                await websocket.send_bytes(audio_data)
                          
            except Exception as e:
                await websocket.send_text(f"Error: {str(e)}")

    except WebSocketDisconnect:
        chat_hist = []
        print("WebSocket got disconnected")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)