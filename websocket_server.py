import asyncio
import websockets
import json
from pathlib import Path
from typing import List

import config
from config import MessageType, StatusType, StatsType
from utils.logger import logger
from utils.recording_saver import RecordingSaver

from core.model.vllm_inference import QwenSession
from core.conversation.conversation_manager import ConversationManager
from core.rag.rag_manager import RAGManager
from core.functions import FunctionExecutor, FunctionHandler


class MessageBuilder:
    @staticmethod
    def status(status: str) -> str:
        return json.dumps({"type": MessageType.STATUS, "status": status})

    @staticmethod
    def text_token(text: str, final: bool = False) -> str:
        return json.dumps({"type": MessageType.TEXT, "text": text, "final": final})

    @staticmethod
    def stats_first_token(time: float) -> str:
        return json.dumps({"type": MessageType.STATS, "stat": StatsType.FIRST_TOKEN, "time": time})

    @staticmethod
    def stats_complete(tokens: int, time: float, tok_per_sec: float) -> str:
        return json.dumps({
            "type": MessageType.STATS,
            "stat": StatsType.COMPLETE,
            "tokens": tokens,
            "time": time,
            "tok_per_sec": tok_per_sec
        })

    @staticmethod
    def error(error: str) -> str:
        return json.dumps({"type": MessageType.ERROR, "error": error})

    @staticmethod
    def pong() -> str:
        return json.dumps({"type": MessageType.PONG})

    @staticmethod
    def identity_info(identity_ids: List[str], profiles: List[dict]) -> str:
        return json.dumps({
            "type": "identity",
            "identity_ids": identity_ids,
            "profiles": profiles
        })


class AnnieMieServer:
    def __init__(self, host=None, port=None):
        self.host = host or config.HOST
        self.port = port or config.PORT
        self.qwen_session = None
        self.conversation = ConversationManager()
        self.rag_manager = RAGManager()
        self.function_executor = None
        self.recording_saver = RecordingSaver()
        self.clients = set()
    
    async def initialize(self):
        logger.header("Initializing AnnieMie Server")

        logger.info("Loading conversation history...")
        self.conversation.load()

        logger.info("Initializing RAG Manager...")
        self.rag_manager.initialize()

        logger.info("Initializing Function Executor...")
        self.function_executor = FunctionExecutor(self.rag_manager)

        logger.info("Initializing Qwen session...")
        self.qwen_session = QwenSession(config.QWEN_MODEL_ID)
        self.qwen_session.initialize()

        logger.success("Server initialized!")
    
    async def handle_client(self, websocket):
        self.clients.add(websocket)
        client_addr = websocket.remote_address
        logger.info(f"Client connected: {client_addr}")
        
        try:
            async for message in websocket:
                await self.process_message(websocket, message)
        
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client disconnected: {client_addr}")
        
        finally:
            self.clients.remove(websocket)
    
    async def process_message(self, websocket, message):
        try:
            data = json.loads(message)
            msg_type = data.get("type")
            
            if msg_type == MessageType.AUDIO:
                await self.handle_audio(websocket, data)

            elif msg_type == MessageType.PING:
                await websocket.send(MessageBuilder.pong())
            
            else:
                logger.warning(f"Unknown message type: {msg_type}")
        
        except json.JSONDecodeError:
            logger.error("Invalid JSON received from client")
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            import traceback
            traceback.print_exc()
    
    async def handle_audio(self, websocket, data):
        audio_base64 = data.get("audio_base64")
        audio_format = data.get("audio_format", "wav")
        session_name = data.get("session_name")
        identity_ids = data.get("identity_ids", [])
        new_identity_ids = data.get("new_identity_ids", [])
        video_frames_b64 = data.get("video_frames", [])

        if not audio_base64 or not session_name:
            logger.error("Missing audio data or session name")
            return

        logger.info(f"Processing session: {session_name}")

        self.recording_saver.create_session_folder(session_name)

        audio_path = self.recording_saver.save_audio_from_base64(audio_base64, audio_format)

        if video_frames_b64:
            for i, frame_b64 in enumerate(video_frames_b64):
                self.recording_saver.save_frame_from_base64(frame_b64, i)
            logger.info(f"✓ {len(video_frames_b64)} video frames saved")

        if identity_ids:
            logger.info(f"Detected identities: {identity_ids}")
            if new_identity_ids:
                logger.info(f"New identities: {new_identity_ids}")
        else:
            identity_ids = ["id-unknown"]
            logger.warning("No identity detected, using unknown")

        identity_contexts = self.rag_manager.build_multi_identity_context(identity_ids)
        
        self.conversation.set_identity_context(identity_contexts)
        
        profiles = [ctx for ctx in identity_contexts]
        await websocket.send(MessageBuilder.identity_info(identity_ids, profiles))

        primary_id = identity_ids[0] if identity_ids else "id-unknown"
        self.function_executor.set_identity(primary_id)

        self.conversation.process_audio(
            audio_path=audio_path,
            session_folder=str(self.recording_saver.current_session_dir),
            identity_ids=identity_ids
        )

        logger.info("Generating response...")
        await websocket.send(MessageBuilder.status(StatusType.GENERATING))

        identity_context_str = self.conversation.get_identity_context()

        final_response = await self._generate_with_functions(
            websocket=websocket,
            identity_context=identity_context_str,
            max_function_iterations=3
        )

        self.conversation.add_assistant_response(final_response)

        logger.success("Response sent to client")
        logger.separator()

    async def _generate_with_functions(
        self,
        websocket,
        identity_context: str = None,
        max_function_iterations: int = 3
    ) -> str:
        iteration = 0
        accumulated_response = ""

        try:
            while iteration < max_function_iterations:
                iteration += 1

                response = await self._stream_generation(
                    websocket=websocket,
                    identity_context=identity_context,
                    is_continuation=(iteration > 1)
                )

                if FunctionHandler.has_function_calls(response):
                    logger.info(f"Function calls detected in response (iteration {iteration})")
                    
                    function_calls = FunctionHandler.parse_function_calls(response)
                    
                    function_results = []
                    for call in function_calls:
                        is_valid, error = FunctionHandler.validate_function_call(call)
                        
                        if is_valid:
                            result = self.function_executor.execute(
                                call["name"],
                                call["arguments"]
                            )
                            formatted_result = FunctionHandler.format_function_result(
                                call["name"],
                                result
                            )
                            function_results.append(formatted_result)
                            logger.info(f"Executed function: {call['name']}")
                        else:
                            function_results.append(f"[Function error: {error}]")
                            logger.warning(f"Invalid function call: {error}")

                    clean_response = FunctionHandler.remove_function_calls(response)
                    accumulated_response += clean_response

                    if function_results:
                        results_text = "\n".join(function_results)
                        self.conversation.add_system_note(f"Function results:\n{results_text}")

                else:
                    accumulated_response += response
                    break

        except Exception as e:
            logger.error(f"Generation error: {e}")
            accumulated_response = f"[Error: {str(e)}]"
        
        finally:
            # ALWAYS send completion signals no matter what
            try:
                logger.debug("📤 Sending final=True message")
                await websocket.send(MessageBuilder.text_token("", final=True))
                logger.debug("📤 Sending status=done message")
                await websocket.send(MessageBuilder.status(StatusType.DONE))
                logger.debug("✅ Completion signals sent successfully")
            except Exception as e:
                logger.error(f"Failed to send completion signals: {e}")

        return accumulated_response.strip()

    async def _stream_generation(
        self,
        websocket,
        identity_context: str = None,
        is_continuation: bool = False
    ) -> str:
        first_token_sent = False
        pending_sends = []  # Track pending send tasks
        
        async def send_token(token: str):
            msg = MessageBuilder.text_token(token)
            logger.debug(f"📤 WS SEND token: {repr(token[:50] if len(token) > 50 else token)}")
            try:
                await websocket.send(msg)
            except Exception as e:
                logger.error(f"❌ Failed to send token: {e}")
        
        async def send_stats(stat_type: str, data):
            if stat_type == "first_token":
                msg = MessageBuilder.stats_first_token(data)
                logger.debug(f"📤 WS SEND first_token: {data}")
            elif stat_type == "complete":
                msg = MessageBuilder.stats_complete(
                    data["tokens"],
                    data["time"],
                    data["tok_per_sec"]
                )
                logger.debug(f"📤 WS SEND complete: {data}")
            elif stat_type == "error":
                msg = MessageBuilder.error(data)
                logger.debug(f"📤 WS SEND error: {data}")
            else:
                return
            try:
                await websocket.send(msg)
            except Exception as e:
                logger.error(f"❌ Failed to send stats: {e}")
        
        # Sync wrappers for callbacks (called from sync code path in generate)
        def token_callback(token: str):
            loop = asyncio.get_event_loop()
            task = loop.create_task(send_token(token))
            pending_sends.append(task)
        
        def stats_callback(stat_type: str, data):
            loop = asyncio.get_event_loop()
            task = loop.create_task(send_stats(stat_type, data))
            pending_sends.append(task)

        try:
            response = await self.qwen_session.generate_async(
                self.conversation.get_history(),
                token_callback,
                stats_callback,
                identity_context
            )
            
            # Wait for all pending sends to complete
            if pending_sends:
                logger.debug(f"⏳ Waiting for {len(pending_sends)} pending sends...")
                await asyncio.gather(*pending_sends, return_exceptions=True)
                logger.debug(f"✅ All sends completed")

            return response

        except Exception as e:
            logger.error(f"Error during generation: {e}")
            await websocket.send(MessageBuilder.error(str(e)))
            raise

    async def start(self):
        await self.initialize()

        logger.header(f"Starting WebSocket Server on {self.host}:{self.port}")

        async with websockets.serve(
            self.handle_client,
            self.host,
            self.port,
            max_size=10*1024*1024,
            ping_interval=20,
            ping_timeout=60
        ):
            logger.success(f"Server listening on ws://{self.host}:{self.port}")
            logger.info("Waiting for clients...")

            await asyncio.Future()

