import json
import logging
import uuid
from fastapi import APIRouter, WebSocket

from aichat.types import MESSAGE_TYPE_SDP_ANSWER, MESSAGE_TYPE_SDP_OFFER
from .manager import ConnectionManager

negotiator = APIRouter(prefix="/ws")

manager = ConnectionManager()

@negotiator.websocket("/sdp")
async def websocket(ws: WebSocket):
    await ws.accept()

    id = uuid.uuid4()
    try:
        async for message in ws.iter_text():
            data = json.loads(message)
            
            if data['type'] == MESSAGE_TYPE_SDP_OFFER:
                await manager.register(id, ws)
                logging.info("accepting sdp offer for connection %s ...", id)
                answer = await manager.accept_offer(id, data['sdp'])
                await ws.send_json({
                    "type": MESSAGE_TYPE_SDP_ANSWER,
                    "sdp": answer,
                }) 
                continue 
                
    except Exception as e:
        logging.error(e)
    
    finally:
        await manager.remove_rtc(id)