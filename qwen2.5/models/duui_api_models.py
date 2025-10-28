from enum import Enum
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from typing import List, Optional


class MultiModelModes(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    FRAMES = "frames"
    VIDEO = "video"
    FRAMES_AND_AUDIO = "frames_and_audio"



class Settings(BaseSettings):
    # Name of this annotator
    mm_annotator_name: str
    # Version of this annotator
    # TODO add these to the settings
    mm_annotator_version: str
    # Log level
    mm_log_level: str
    # # # model_name
    # Name of this annotator
    mm_model_version: str
    #cach_size
    mm_model_cache_size: str



# Documentation response
class DUUIMMDocumentation(BaseModel):
    # Name of this annotator
    annotator_name: str

    # Version of this annotator
    version: str

    # Annotator implementation language (Python, Java, ...)
    implementation_lang: Optional[str]

    # Optional map of additional meta data
    meta: Optional[dict]

    # Docker container id, if any
    docker_container_id: Optional[str]

    # Optional map of supported parameters
    parameters: Optional[dict]


class ImageType(BaseModel):
    """
    org.texttechnologylab.annotation.type.Image
    """
    src: str
    width: int
    height: int
    begin: int
    end: int

class Entity(BaseModel):
    """
    Named bounding box entity
    name: entity name
    begin: start position
    end: end position
    bounding_box: list of bounding box coordinates
    """
    name: str
    begin: int
    end: int
    bounding_box: List[tuple[float, float, float, float]]

class LLMMessage(BaseModel):
    role: str = None
    content: str
    class_module: str = None
    class_name: str = None
    fillable: bool = False
    context_name: str = None
    ref: int   # internal cas annotation id


class LLMPrompt(BaseModel):
    messages: List[LLMMessage]
    args: Optional[str]  # json string
    ref: Optional[int]   # internal cas annotation id

class LLMResult(BaseModel):
    meta: str  # json string
    prompt_ref: int   # internal cas annotation id
    message_ref: str   # internal cas annotation id

class VideoTypes(BaseModel):
    """
    org.texttechnologylab.annotation.type.Video
    """
    src: str
    length: int = -1
    fps: int = -1
    begin: int
    end: int


class AudioType(BaseModel):
    """
    org.texttechnologylab.annotation.type.Audio
    """
    src: str
    begin: int
    end: int
      
# Request sent by DUUI
# Note, this is transformed by the Lua script
class DUUIMMRequest(BaseModel):

    # list of images
    images: Optional[List[ImageType]]
    # audio
    audios: Optional[List[AudioType]]

    # videos
    videos :Optional[List[VideoTypes]]

    # List of prompt
    prompts: List[LLMPrompt]

    # doc info
    doc_lang: str

    # model name
    model_name: str

    # individual or multiple image processing
    individual: bool = False

    # mode for complex
    mode: MultiModelModes = MultiModelModes.TEXT






# Response sent by DUUI
# Note, this is transformed by the Lua script
class DUUIMMResponse(BaseModel):
    # list of processed text
    processed_text: Optional[List[LLMResult]]

    # model source
    model_source: str
    # model language
    model_lang: str
    # model version
    model_version: str
    # model name
    model_name: str
    # list of errors
    errors: Optional[List[LLMResult]]

    # original prompt
    prompts: List[Optional[LLMPrompt]] = []