from llama_index.core.workflow import (
    StartEvent,
)

from app.agents.chat_history import ChatHistory
from app.agents.types import InitialChatEvent


def start_to_input(ev: StartEvent, history: ChatHistory) -> InitialChatEvent:
    # get user input
    for msg in ev.chat_history:
        history.add(msg)

    return InitialChatEvent()
