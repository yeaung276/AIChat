import json
import logging
import uuid
from fastapi import APIRouter, WebSocket

from .manager import RTCManager

negotiator = APIRouter(prefix="/ws")

manager = RTCManager()

@negotiator.websocket("/sdp")
async def websocket(ws: WebSocket):
    await ws.accept()

    id = uuid.uuid4()
    
    try:
        await manager.create_rtc(id)
        async for message in ws.iter_text():
            data = json.loads(message)
            
            if data['type'] == "offer":
                logging.info("accepting sdp offer for connection %s ...", id)
                answer = await manager.accept_offer(id, data['sdp'])
                await ws.send_json({
                    "type": "answer",
                    "sdp": answer,
                }) 
                continue 
                
    except Exception as e:
        logging.error(e)
    
    finally:
        await manager.remove_rtc(id)