from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage , SystemMessage
from langgraph.graph.message import add_messages
import os
from langgraph.prebuilt import ToolNode, tools_condition
from typing import TypedDict, Annotated
from langgraph.checkpoint.memory import MemorySaver
from tools import rem_null_duplicates,data_profile_tool, kpi_summary_tool,correlation_tool,encode_categorical_tool,groupby_summary_tool, outlier_detection_tool, plot_distribution_tool, plot_correlation_heatmap_tool ,preprocess_dates_tool, prediction
from registry import DATASET_REGISTRY, model


load_dotenv()
api_key=os.getenv('API_KEY')

llm=ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=api_key
)

#defining state
class AutoML(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def load_dataset(dataset_id: str, path: str):
    """Load dataset and store internally."""
    DATASET_REGISTRY[dataset_id] = pd.read_csv(path)


tools=[rem_null_duplicates,data_profile_tool, kpi_summary_tool,correlation_tool,encode_categorical_tool,groupby_summary_tool, outlier_detection_tool, plot_distribution_tool, plot_correlation_heatmap_tool ,preprocess_dates_tool, prediction]


llm_tools=llm.bind_tools(tools)
SYSTEM_INSTRUCTIONS=SystemMessage(
    content="""
   You are an expert Machine Learning assistant operating in a tool-driven environment.

Your responsibilities:
- Select and invoke the most appropriate available tool and model according to the provided data such that maximum accuracy is obtained.

Constraints:
- Do NOT infer, estimate, or fabricate results.
- Do NOT perform calculations, predictions, or analysis without using a tool.
- If a required capability or tool is not available, clearly respond:
  "I cannot perform this task with the current tools."

Behavior rules:
- Base all responses only on tool outputs or explicit user-provided information.
- Be concise, factual, and deterministic.
- Never hallucinate missing data, metrics, or outcomes.

"""
)

checkpointer= MemorySaver()

def chat_node(state: AutoML):

    messages = state["messages"]

    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SYSTEM_INSTRUCTIONS] + messages

    response = llm_tools.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)  

graph=StateGraph(AutoML)
graph.add_node("LLM",chat_node)

graph.add_node("tools",tool_node)

graph.add_edge(START, "LLM")
graph.add_conditional_edges("LLM",tools_condition)
graph.add_edge("tools","LLM")


workflow=graph.compile(checkpointer=checkpointer)

def load_dataset(dataset_id: str, df: pd.DataFrame):
    DATASET_REGISTRY[dataset_id] = df

def run(task: str, thread_id: str = "thread1"):
    result = workflow.invoke(
        {
            "messages": [HumanMessage(content=task)]
        },
        config={
            "configurable": {
                "thread_id": thread_id
            }
        }
    )
    return result["messages"][-1].content
