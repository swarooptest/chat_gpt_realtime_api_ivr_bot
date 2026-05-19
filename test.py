import os
import json
import base64
import asyncio
import websockets
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.websockets import WebSocketDisconnect
from twilio.twiml.voice_response import VoiceResponse, Connect, Say, Stream
from dotenv import load_dotenv

load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PORT = int(os.getenv('PORT', 5050))
SYSTEM_MESSAGE = (
    "You are a Parenting Guide bot of Britannia Milk Bikis, designed to assist users with advice about parenting, "
    "focusing on children’s nutrition and providing information about various scholarships run by the Government of Uttar Pradesh. "
    "You have a detailed knowledge base specifically about the scholarships of Uttar Pradesh as stated in the user corpus."
    "\nGuidelines for Responses:"
    "\n1. Respond in Hindi: All responses must be in Hindi and strictly relevant to the user’s query."
    "\n2. Personalize the Response: Use the child’s name in every response once the user provides it."
    "\n3. Conciseness: Limit responses to 50-60 words maximum, keeping them polite and informative."
    "\n4. End with a Britannia Message based on context."
    "\n5. Ask a Follow-Up Question to ensure continuity in the conversation."
    "\nSequence of the Call:"
    "\n1. Welcome Message: ब्रिटानिया मिल्क बिकिस परवरिश की बात में आपका स्वागत है। मैं पंकज त्रिपाठी हूं - वैसे आप मुझे पापा त्रिपाठी कह सकते हैं। "
    "तो बताइये आज पेरेंटिंग के बारे में क्या जानना चाहेंगे? मुझे बताइये आप का बेटा है या बेटी?"
    "\n2. User Input: User mentions Boy or Girl."
    "\n3. Bot asks: उसका नाम क्या है?"
    "\n4. User Input: Child’s name."
    "\n5. Bot asks: <Child’s Name> कौन सी कक्षा में है?"
    "\n6. User Input: User specifies the child’s Standard/Class."
    "\n7. Bot asks: क्या आप उन विभिन्न छात्रवृत्तियों, वज़ीफ़ों के बारे में जानते हैं जिनका लाभ <Child’s Name> उत्तर प्रदेश सरकार से ले सकता है?"
    "\nStrictly adhere to this flow."
)

VOICE = 'alloy'
LOG_EVENT_TYPES = ['error', 'response.content.done', 'rate_limits.updated', 'response.done']
SHOW_TIMING_MATH = False

app = FastAPI()

if not OPENAI_API_KEY:
    raise ValueError('Missing the OpenAI API key. Please set it in the .env file.')

@app.get("/", response_class=JSONResponse)
async def index_page():
    return {"message": "Twilio Media Stream Server is running!"}


@app.api_route("/incoming-call", methods=["GET", "POST"])
async def handle_incoming_call(request: Request):
    """Handle incoming call and return TwiML response to connect to Media Stream."""
    response = VoiceResponse()
    response.say(
        "ब्रिटानिया मिल्क बिकिस परवरिश की बात में आपका स्वागत है। मैं पंकज त्रिपाठी हूं - वैसे आप मुझे पापा त्रिपाठी कह सकते हैं। "
        "तो बताइये आज पेरेंटिंग के बारे में क्या जानना चाहेंगे? मुझे बताइये आप का बेटा है या बेटी?"
    )
    host = request.url.hostname
    connect = Connect()
    connect.stream(url=f'wss://{host}/media-stream')
    response.append(connect)
    return HTMLResponse(content=str(response), media_type="application/xml")


@app.websocket("/media-stream")
async def handle_media_stream(websocket: WebSocket):
    """Handle WebSocket connections between Twilio and OpenAI."""
    print("Client connected")
    await websocket.accept()

    async with websockets.connect(
            'wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01',
            extra_headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "OpenAI-Beta": "realtime=v1"
            }
    ) as openai_ws:
        await initialize_session(openai_ws)

        async def receive_from_twilio():
            """Receive audio data from Twilio and send it to the OpenAI Realtime API."""
            try:
                async for message in websocket.iter_text():
                    data = json.loads(message)
                    if data['event'] == 'media' and openai_ws.open:
                        audio_append = {
                            "type": "input_audio_buffer.append",
                            "audio": data['media']['payload']
                        }
                        await openai_ws.send(json.dumps(audio_append))
            except WebSocketDisconnect:
                print("Client disconnected.")
                if openai_ws.open:
                    await openai_ws.close()

        async def send_to_twilio():
            """Receive events from the OpenAI Realtime API and send audio back to Twilio."""
            try:
                async for openai_message in openai_ws:
                    response = json.loads(openai_message)
                    if response.get('type') == 'response.audio.delta' and 'delta' in response:
                        audio_payload = base64.b64encode(base64.b64decode(response['delta'])).decode('utf-8')
                        audio_delta = {
                            "event": "media",
                            "streamSid": response['streamSid'],
                            "media": {
                                "payload": audio_payload
                            }
                        }
                        await websocket.send_json(audio_delta)
            except Exception as e:
                print(f"Error in send_to_twilio: {e}")

        await asyncio.gather(receive_from_twilio(), send_to_twilio())


async def initialize_session(openai_ws):
    """Initialize the session with OpenAI Realtime API."""
    session_update = {
        "type": "session.update",
        "session": {
            "turn_detection": {"type": "server_vad"},
            "input_audio_format": "g711_ulaw",
            "output_audio_format": "g711_ulaw",
            "voice": VOICE,
            "instructions": SYSTEM_MESSAGE,
            "modalities": ["text", "audio"],
            "temperature": 0.8,
        }
    }
    await openai_ws.send(json.dumps(session_update))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)
