import io

import soundfile as sf
from pydantic import BaseModel, ConfigDict


class AudioData(BaseModel):
    bytes: bytes
    path: str

    def load(self):
        data, sr = sf.read(io.BytesIO(self.bytes))
        return data, sr


class LibriSpeechSample(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    file: str
    audio: AudioData
    text: str
    speaker_id: int
    chapter_id: int
    id: str
