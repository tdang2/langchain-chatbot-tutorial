import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage


# Load environment variables from .env file
load_dotenv()

def main():
    model = init_chat_model("claude-3-5-haiku-20241022", model_provider="anthropic")
    messages = [
        SystemMessage("Translate the following from English into Vietnamese"),
        HumanMessage("Hello! My name is Miro"),
    ]
    result_messages = model.invoke(messages)
    print(result_messages.pretty_print())

if __name__ == "__main__":
    main()
