from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum

class RequestState(str, Enum):
    WAITING="waiting"; RUNNING="running"; FINISHED="finished"; FAILED="failed"; CANCELLED="cancelled"

@dataclass
class SamplingParams:
    max_new_tokens:int=64
    temperature:float=0.0
    top_p:float=1.0
    top_k:int|None=None
    min_p:float=0.0
    repetition_penalty:float=1.0
    seed:int|None=None
    stop_token_ids:tuple[int,...]=()

@dataclass
class Request:
    request_id:str
    prompt_token_ids:tuple[int,...]
    sampling:SamplingParams=field(default_factory=SamplingParams)
    state:RequestState=RequestState.WAITING
    generated_token_ids:list[int]=field(default_factory=list)
    prompt_cursor:int=0
    error:str|None=None

    @property
    def done(self)->bool:
        return self.state in {RequestState.FINISHED,RequestState.FAILED,RequestState.CANCELLED}
