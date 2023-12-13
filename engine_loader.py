from llama_index import StorageContext, load_index_from_storage
import os
from llama_index import (
  StorageContext,
  load_index_from_storage
  )

from llama_index.llms import AzureOpenAI
from llama_index.embeddings import AzureOpenAIEmbedding
from llama_index import ServiceContext
from llama_index import set_global_service_context


llm = AzureOpenAI(
    model="gpt-35-turbo",
    deployment_name="gpt-3",
    api_key=os.environ["OPENAI_API_KEY"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_version=os.environ['OPENAI_API_VERSION'],
)
embed_model = AzureOpenAIEmbedding(
    model="text-embedding-ada-002",
    deployment_name=os.environ['EMBEDDING_DEPLOYMENT_NAME'],
    api_key=os.environ['OPENAI_API_KEY'],
    azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT'],
    api_version=os.environ['OPENAI_API_VERSION'],
)

def __get_index(persist_dir: str):
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    index = load_index_from_storage(storage_context)
    return index

def get_query_engines(): 
    service_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
    )

    set_global_service_context(service_context)

    ipc_act_index = __get_index("./vector_store/ipc")
    nyay_index = __get_index("./vector_store/bns")

    ipc_act_engine = ipc_act_index.as_chat_engine(verbose=True)
    nyay_engine = nyay_index.as_chat_engine(verbose=True)

    # query_engine_tool = [
    #     QueryEngineTool(
    #         query_engine=ipc_act_engine,
    #         metadata=ToolMetadata(
    #             name="Indian Penal Code (IPC) Act Query Engine",
    #             description="Provides Information about Indian Penal Code(IPC) system.",
    #         ),
    #     ),
    #     QueryEngineTool(
    #         query_engine=nyay_engine,
    #         metadata=ToolMetadata(
    #             name="Bharatiya Nyaya Sanhita (BNS) Query Engine",
    #             description="Provides Information about BHARATIYA NYAYA SANHITA system.",
    #         ),
    #     )
    # ]

    # query_engine = SubQuestionQueryEngine.from_defaults(query_engine_tools=query_engine_tool,service_context=service_context)

    return ipc_act_engine, nyay_engine
