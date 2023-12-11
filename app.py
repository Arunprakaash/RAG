import os
import chainlit as cl
from engine_loader import get_query_engines
from langchain.agents import Tool, AgentExecutor
from langchain.chat_models import AzureChatOpenAI
from constants import PROMPT
from langchain.memory import ConversationBufferMemory
from langchain.agents import OpenAIFunctionsAgent

os.environ["OPENAI_API_TYPE"]="azure"

llm = AzureChatOpenAI(
    azure_deployment=os.environ["AZURE_DEPLOYMENT"],
    openai_api_version=os.environ['OPENAI_API_VERSION'],
)

try:
    ipc_engine, bns_engine = get_query_engines()
except:
   print("no vectors")

def ipc_query(query:str)-> str:
        ipc_engine = cl.user_session.get("ipc_query_engine")
        response = str(ipc_engine.query(query))
        return response

def bns_query(query:str)-> str:
    bns_engine = cl.user_session.get("bns_query_engine")
    response = str(bns_engine.query(query))
    return response

@cl.on_chat_start
async def factory():
    cl.user_session.set("ipc_query_engine", ipc_engine)
    cl.user_session.set("bns_query_engine",bns_engine)

    tools = [
        Tool(
            name="BNS-tool",
            func=bns_query,
            description="Useful for when you need find information about Bhartiya Nyaya Sanhita Clauses(BNS) (Bill Replacing IPC) .",
        ),
        Tool(
            name="IPC-tool",
            func=ipc_query,
            description="Useful for when you need find information about Indian Penal Section Codes(IPC).",
        )
    ]

    memory = ConversationBufferMemory(memory_key="memory", return_messages=True)

    
    agent = OpenAIFunctionsAgent(
        llm=llm,
        tools=tools,
        prompt= PROMPT,
        
    )

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        max_iterations=30
    )

    cl.user_session.set("lamarr_ai",agent_executor)


@cl.on_message
async def main(message: cl.Message):
    lamarr_ai = cl.user_session.get("lamarr_ai")

    response = await cl.make_async(lamarr_ai.run)(message.content)
    await cl.Message(content=response).send()
    