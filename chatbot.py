from typing import Annotated, Any
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

load_dotenv()

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7,
)

def call_model(state: State) -> dict[str, Any]:
    response = model.invoke(state["messages"])
    return {"messages": response}

graph = StateGraph(State)
graph.add_node("call_model", call_model)
graph.add_edge(START, "call_model")
graph.add_edge("call_model", END)

app = graph.compile()

def main() -> None:
    print("🤖 Groq Chatbot is ready! (Type 'quit' to exit)")
    messages: list[AnyMessage] = []

    while True:
        user_input = input("\n👤 You: ").strip()
        if user_input.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break
        if not user_input:
            continue

        messages.append(HumanMessage(content=user_input))
        result = app.invoke({"messages": messages})
        messages = result["messages"]

        ai_msg = messages[-1]
        print(f"\n🤖 Bot: {ai_msg.content}")

if __name__ == "__main__":
    main()