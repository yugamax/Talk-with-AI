import os
import tempfile
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
import uvicorn
from dotenv import load_dotenv

load_dotenv()
client1 = Groq(api_key=os.getenv("gr_api_key1"))
client2 = Groq(api_key=os.getenv("gr_api_key2"))
client3 = Groq(api_key=os.getenv("gr_api_key3"))
client4 = Groq(api_key=os.getenv("gr_api_key4"))
client5 = Groq(api_key=os.getenv("gr_api_key5"))
client6 = Groq(api_key=os.getenv("gr_api_key6"))

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
        "content": "You are a live speaking MedBot named Dr. Groq. You interact with users just like a friendly person would on a phone call! You are warm, lively, empathetic, and knowledgeable, providing clear, friendly, and accurate medical information. Always encourage users to consult real healthcare professionals for any serious or personal medical concerns. Begin each fresh conversation with a very short, cheerful introduction. Keep all responses brief — ideally under 80 words — and break longer explanations into multiple short responses if needed. Speak naturally and expressively, using a positive tone and lively punctuation like \"!\", \"...\", and \":\". Always explain medical terms simply, using easy-to-understand language. Sound supportive, caring, and professional at all times. You must only answer questions related to healthcare, medicine, wellness, or medical education. If a user asks anything outside of these topics, kindly reply: \'I focus only on health-related topics! Let’s chat about anything health or wellness you need help with. 🌟\' Never give direct diagnoses, treatment plans, or prescriptions. If a question goes beyond your capability, kindly suggest: \' It’s best to talk to a licensed healthcare professional for that! 💬\' Your primary goal is to make users feel heard, cared for, and guided at every step."
    }]
    
    try:
        while True:
            audio_bytes = await websocket.receive_bytes()

            with tempfile.NamedTemporaryFile(delete=False, suffix=".m4a") as temp_audio:
                temp_audio.write(audio_bytes)
                temp_audio_path = temp_audio.name

            with open(temp_audio_path, "rb") as file:
                transcription = client6.audio.transcriptions.create(
                    file=(temp_audio_path, file.read()),
                    model="whisper-large-v3-turbo",
                    response_format="verbose_json",
                )
            os.remove(temp_audio_path)

            chat_hist.append({"role": "user", "content": transcription.text})

            try:
                completion = client6.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=chat_hist,
                    temperature=0.2,
                    max_tokens=256,
                )
                res = completion.choices[0].message.content
                chat_hist.append({"role": "assistant", "content": res})

                clients = [client1, client2, client3, client4, client5, client6]
                for i, client in enumerate(clients, 1):
                    try:
                        print(f"Handling client {i} and it's id :{client}")
                        response = client.audio.speech.create(
                                            model="playai-tts",
                                            voice="Nia-PlayAI",
                                            response_format="wav",
                                            input=res
                                        )
                        break
                    except Exception as e:
                        print(f"Client {i} failed: {e}")
                else:
                    await websocket.send_text("All TTS models unavailable.")
                    return
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
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)