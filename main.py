import os
from dotenv import load_dotenv
from typing import Annotated, TypedDict
from IPython.display import Image, display

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults

from base_tool_node import BasicToolNode



# Load environment variables from .env file
load_dotenv()


class State(TypedDict):
     # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]


def main():
    def chatbot(state: State):
        return {
            "messages": [llm_with_tool.invoke(state["messages"])]
        }
    
    graph_builder = StateGraph(State)
    llm = ChatAnthropic(model="claude-3-5-haiku-20241022")
    tool = TavilySearchResults(max_results=2)
    tools = [tool]
    llm_with_tool = llm.bind_tools(tools)
    # The first argument is the unique node name
    # The second argument is the function or object that will be called whenever
    # the node is used.
    graph_builder.add_node("chatbot", chatbot)
    graph_builder.set_entry_point("chatbot")
    tool_node = ToolNode(tools=[tool])
    graph_builder.add_node("tools", tool_node)
    # The `tools_condition` function returns "tools" if the chatbot asks to use a tool, and "END" if
    # it is fine directly responding. This conditional routing defines the main agent loop.
    graph_builder.add_conditional_edges(
        "chatbot",
        tools_condition
    )
    # Any time a tool is called, we return to the chatbot to decide the next step
    graph_builder.add_edge("tools", "chatbot")
    graph = graph_builder.compile()
    
    def stream_graph_updates(user_input: str):
        for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
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
    
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            stream_graph_updates(user_input)
        except:
            # fallback if input() is not available
            user_input = "What do you know about LangGraph?"
            print("User: " + user_input)
            stream_graph_updates(user_input)
            break
    

if __name__ == "__main__":
    main()
