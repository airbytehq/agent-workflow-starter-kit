from llama_index.core.llms import LLM, ChatMessage
from llama_index.core.memory import ChatMemoryBuffer


class ChatHistory:
    def __init__(self, llm: LLM):
        self.memory = ChatMemoryBuffer.from_defaults(llm=llm)

    def get(self) -> list[ChatMessage]:
        messages = self.memory.get()  # includes the context window chop
        while len(messages) == 0:
            # probably ran out of memory from the last response itself being too large
            # back up a bit to get all but the last message
            all_messages = self.memory.get_all()
            last_message = all_messages[-1]
            less_messages = all_messages[:-1]
            shorter_message = ChatMessage(
                role=last_message.role,
                content="Error: Ran out of memory. This message was too long.",
            )
            self.memory.set(less_messages + [shorter_message])
            messages = self.memory.get()
        return messages

    def add(self, message: ChatMessage) -> None:
        self.memory.put(message)
