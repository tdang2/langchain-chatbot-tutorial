import os
from dotenv import load_dotenv
from typing import Annotated, TypedDict
from IPython.display import Image, display

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.types import Command, interrupt
from langchain.chat_models import init_chat_model
from langchain_core.messages import ToolMessage

from base_tool_node import BasicToolNode



# Load environment variables from .env file
load_dotenv()


class State(TypedDict):
     # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]
    name: str
    birthday: str


def main():
    from langchain_core.tools import tool, InjectedToolCallId
    
    def chatbot(state: State):
        message = llm_with_tools.invoke(state["messages"])
        # Because we will be interrupting during tool execution,
        # we disable parallel tool calling to avoid repating any
        # tool invocations when we resume
        assert len(message.tool_calls) <= 1
        return {
            "messages": [message]
        }
        
    @tool
    # Note that because we are generating a ToolMessage for a state update, we
    # generally require the ID of the corresponding tool call. We can use
    # LangChain's InjectedToolCallId to signal that this argument should not
    # be revealed to the model in the tool's schema.
    def human_assistance(
        name: str, birthday: str, tool_call_id: Annotated[str, InjectedToolCallId]
    ) -> str:
        """Request assistance from a human."""
        human_response = interrupt(
            {
                "question": "Is this correct?",
                "name": name,
                "birthday": birthday,
            },
        )
        # If the information is correct, update the state as-is.
        if human_response.get("correct", "").lower().startswith("y"):
            verified_name = name
            verified_birthday = birthday
            response = "Correct"
        # Otherwise, receive information from the human reviewer.
        else:
            verified_name = human_response.get("name", name)
            verified_birthday = human_response.get("birthday", birthday)
            response = f"Made a correction: {human_response}"

        # This time we explicitly update the state with a ToolMessage inside
        # the tool.
        state_update = {
            "name": verified_name,
            "birthday": verified_birthday,
            "messages": [ToolMessage(response, tool_call_id=tool_call_id)],
        }
        # We return a Command object in the tool to update our state.
        return Command(update=state_update)

    tool = TavilySearchResults(max_results=2)
    tools = [tool, human_assistance]
    llm = init_chat_model("anthropic:claude-3-5-sonnet-latest")
    llm_with_tools = llm.bind_tools(tools)

    graph_builder = StateGraph(State)
    # The first argument is the unique node name
    # The second argument is the function or object that will be called whenever
    # the node is used.
    graph_builder.add_node("chatbot", chatbot)    
    tool_node = ToolNode(tools=tools)
    graph_builder.add_node("tools", tool_node)
    # The `tools_condition` function returns "tools" if the chatbot asks to use a tool, and "END" if
    # it is fine directly responding. This conditional routing defines the main agent loop.
    graph_builder.add_conditional_edges(
        "chatbot",
        tools_condition
    )
    # Any time a tool is called, we return to the chatbot to decide the next step
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.set_entry_point("chatbot")
    # change this to use SqliteSaver or PostgresSaver and connect to your own DB
    memory = MemorySaver()
    graph = graph_builder.compile(checkpointer=memory)
    
    def stream_graph_updates(user_input: str, config: dict):
        for event in graph.stream(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config
        ):
            for value in event.values():
                print("Assistant:", value["messages"][-1].content)
                
    try:
        # Generate the graph visualization as PNG bytes
        graph_png_data = graph.get_graph().draw_mermaid_png()
        img = Image(graph_png_data)
        output_filename = "graph_visualization.png"
        # Save the image data to a file
        with open(output_filename, "wb") as f:
            f.write(img.data) # Access the raw byte data from the Image object
        print(f"Graph visualization saved to {output_filename}")
    except Exception:
        # This requires some extra dependencies and is optional
        pass
    # Learning how to use LLM memory
    # Pick a thread ID to remember the conversation
    config = {
        "configurable": {
            "thread_id": "1"
        }
    }
    user_input = (
        "Can you look up when LangGraph was released? "
        "When you have the answer, use the human_assistance tool for review."
    )
    config = {"configurable": {"thread_id": "1"}}

    events = graph.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config,
        stream_mode="values",
    )
    for event in events:
        if "messages" in event:
            event["messages"][-1].pretty_print()
    human_command = Command(
        resume={
            "name": "LangGraph",
            "birthday": "Jan 17, 2024",
        },
    )

    events = graph.stream(human_command, config, stream_mode="values")
    for event in events:
        if "messages" in event:
            event["messages"][-1].pretty_print()            
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            stream_graph_updates(user_input, config=config)
        except:
            # fallback if input() is not available
            user_input = "What do you know about LangGraph?"
            print("User: " + user_input)
            stream_graph_updates(user_input, config=config)
            break
    

if __name__ == "__main__":
    main()
