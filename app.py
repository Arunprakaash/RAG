import os
import chainlit as cl
from engine_loader import get_query_engines
from langchain.agents import Tool, AgentExecutor
from langchain.agents import OpenAIFunctionsAgent
from langchain.chat_models import AzureChatOpenAI
from constants import PROMPT

os.environ["OPENAI_API_TYPE"]="azure"

llm = AzureChatOpenAI(
    azure_deployment=os.environ["AZURE_DEPLOYMENT"],
    openai_api_version=os.environ['OPENAI_API_VERSION'],
)

try:
    ipc_engine, bns_engine = get_query_engines()
except:
   print("no vectors")

@cl.on_chat_start
async def factory():
    cl.user_session.set("ipc_query_engine", ipc_engine)
    cl.user_session.set("bns_query_engine",bns_engine)


@cl.on_message
async def main(message: cl.Message):
    def ipc_query(query:str)-> str:
        ipc_engine = cl.user_session.get("ipc_query_engine")
        response = str(ipc_engine.query(query))
        return response

    def bns_query(query:str)-> str:
        bns_engine = cl.user_session.get("bns_query_engine")
        response = str(bns_engine.query(query))
        return response

    tools = [
        Tool(
            name="BNS-tool",
            func=bns_query,
            description="""Empower the LangChain agent with the BNS-tool to handle user queries
            related to the official new clauses of the Republic of India.
            Utilize the BHARATIYA NYAYA SANHITA (BNS) system for accurate and up-to-date
            information on Clauses and relevant information. Ensure seamless integration
            with the BNS Query Engine for an enhanced user experience.""",
        ),
        Tool(
            name="IPC-tool",
            func=ipc_query,
            description="""Equip the LangChain agent with the IPC-tool to address user queries
            related to the official old Section codes of the Republic of India.
            Leverage the Indian Penal Code (IPC) system for historical Section codes and relevant information.
            Integrate smoothly with the IPC Act Query Engine to retrieve precise and comprehensive
            information, enriching the user's understanding of the legal framework.""",
        )
    ]

    agent = OpenAIFunctionsAgent(
                llm=llm,
                tools=tools,
                prompt= PROMPT
            )

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=30
    )
    response = await cl.make_async(agent_executor.run)(message.content)
    await cl.Message(content=response).send()
    