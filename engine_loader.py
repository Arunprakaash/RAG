from llama_index import StorageContext, load_index_from_storage
import openai
import os
from llama_index import (
  VectorStoreIndex,
  StorageContext,
  load_index_from_storage
  )
from pathlib import Path

from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.query_engine import SubQuestionQueryEngine
from llama_index.llms import AzureOpenAI

def __get_index(persist_dir: str):
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    index = load_index_from_storage(storage_context)
    return index

def get_query_engines():
    ipc_act_index = __get_index("./vector_store/ipc")
    nyay_index = __get_index("./vector_store/bns")

    ipc_act_engine = ipc_act_index.as_query_engine(similarity_top_k=10)
    nyay_engine = nyay_index.as_query_engine(similarity_top_k=10)

    ipc_act_query_engine = QueryEngineTool(
        query_engine=ipc_act_engine,
        metadata=ToolMetadata(
            name="Indian Penal Code (IPC) Act Query Engine",
            description="""Intelligently understands user queries and searches the Indian Penal Code (IPC) system for relevant Sections and their corresponding description.
            This tool empowers users with comprehensive information, presenting numeric Section codes and detailed descriptions, thereby enhancing accessibility to the IPC legal framework.
            """,
        ),
    )

    bns_query_engine = QueryEngineTool(
        query_engine=nyay_engine,
        metadata=ToolMetadata(
            name="Bharatiya Nyaya Sanhita (BNS) Query Engine",
            description="Empowers users by intelligently understanding their queries and searching the BHARATIYA NYAYA SANHITA (BNS) system for relevant Clauses and their corresponding descriptions. This tool provides users with comprehensive information, including numeric Clause codes and detailed descriptions, enhancing accessibility to the BNS legal framework.",
        ),
    )

    ipc_engine = SubQuestionQueryEngine.from_defaults(query_engine_tools=[ipc_act_query_engine])
    bns_engine = SubQuestionQueryEngine.from_defaults(query_engine_tools=[bns_query_engine])

    return ipc_engine, bns_engine
