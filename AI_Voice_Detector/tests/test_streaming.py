import base64
import json


class FakeWebSocket:
    def __init__(self, messages, api_key):
        self._messages = iter(messages)
        self.sent = []
        self.environ = {"QUERY_STRING": f"api_key={api_key}"}

    def receive(self):
        try:
            return next(self._messages)
        except StopIteration:
            return None

    def send(self, message):
        try:
            self.sent.append(json.loads(message))
        except Exception:
            self.sent.append(message)


def build_pcm16_chunk(sample_rate=16000, channels=1, seconds=1.0):
    bytes_per_second = sample_rate * channels * 2
    size = int(bytes_per_second * seconds)
    return base64.b64encode(b"\x00" * size).decode("utf-8")


def test_streaming_success_with_partial_and_final(app_module):
    start_msg = json.dumps({
        "type": "start",
        "audioFormat": "pcm16",
        "sampleRate": 16000,
        "channels": 1,
        "enablePartial": True,
        "partialIntervalSec": 0.5
    })

    chunk_msg = json.dumps({
        "type": "audio_chunk",
        "audioChunkBase64": build_pcm16_chunk(seconds=1.0),
        "final": True
    })

    ws = FakeWebSocket([start_msg, chunk_msg], api_key=app_module.API_KEY)
    app_module.voice_stream(ws)

    types = [msg.get("type") for msg in ws.sent if isinstance(msg, dict)]
    assert "ack" in types
    assert "progress" in types
    assert "partial_result" in types
    assert "final_result" in types


def test_streaming_invalid_api_key(app_module):
    start_msg = json.dumps({
        "type": "start",
        "audioFormat": "pcm16",
        "sampleRate": 16000,
        "channels": 1
    })

    ws = FakeWebSocket([start_msg], api_key="bad_key")
    app_module.voice_stream(ws)

    assert ws.sent
    assert ws.sent[0]["type"] == "error"


def test_streaming_invalid_format(app_module):
    start_msg = json.dumps({
        "type": "start",
        "audioFormat": "aac",
        "sampleRate": 16000,
        "channels": 1
    })

    ws = FakeWebSocket([start_msg], api_key=app_module.API_KEY)
    app_module.voice_stream(ws)

    assert ws.sent
    assert ws.sent[0]["type"] == "error"


def test_streaming_disabled(app_factory):
    app_module = app_factory(ENABLE_STREAMING="false")
    ws = FakeWebSocket([json.dumps({"type": "start"})], api_key=app_module.API_KEY)
    app_module.voice_stream(ws)

    assert ws.sent
    assert ws.sent[0]["type"] == "error"
