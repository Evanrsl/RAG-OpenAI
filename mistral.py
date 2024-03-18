import openai_example
from pinecone import Pinecone as pinecone

import os
from dotenv import load_dotenv

from langchain_community.chat_models import ChatAnyscale
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.schema.runnable.config import RunnableConfig
from langchain.memory import ConversationBufferMemory

from operator import itemgetter
import chainlit as cl


# load .env file
load_dotenv()
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME')
ANYSCALE_API_BASE = os.getenv('ANYSCALE_API_BASE')
ANYSCALE_API_KEY = os.getenv('ANYSCALE_API_KEY')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL')
LLM_MODEL = os.getenv('LLM_MODEL')

# initialize connection to pinecone (get API key at app.pinecone.io)
# configure client
pc = pinecone(api_key=PINECONE_API_KEY)

# Connect to index mistral-rag
index_name = PINECONE_INDEX_NAME
existing_indexes = [
    index_info["name"] for index_info in pc.list_indexes()
]

# check if index exists
if index_name in existing_indexes:
    # connect to index
    index = pc.Index(index_name)
    # view index stats
    print(index.describe_index_stats())
else:
    print("index not exist")


# Initialize Embeddings model
def embedding(query):
    client = openai_example.OpenAI(
        base_url=ANYSCALE_API_BASE,
        api_key=ANYSCALE_API_KEY
    )
    embed = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
    return (embed.model_dump()['data'][0]['embedding'])


def augment_prompt(prompt: str):
    # get top 3 results from knowledge base
    results = index.query(
        vector=embedding(prompt['question']),
        top_k=5,
        include_metadata=True
    )

    source_knowledge = ""
    num_of_context = 0
    for result in results['matches']:
        score = result['score']
        text = result['metadata']['text']
        print(score)
        if score  > 0.85 and len(text) > 50:
            print(text)
            source_knowledge += f"text: {text} \n source:{result['metadata']['source']} \n\n"
            num_of_context += 1
    print(num_of_context)
    # feed into an augmented prompt
    if num_of_context > 0 :
        augmented_prompt = f"""{prompt['question']}

        gunakan informasi ini jika berguna:
        {source_knowledge}"""
        print(augmented_prompt)
        return {"question": f"{augmented_prompt}"}
    return prompt


# Chainlit

@cl.on_chat_start
async def on_chat_start():
    memory = ConversationBufferMemory(return_messages=True)

    # Initialize the model with specific configurations
    model = ChatOpenAI(streaming=True)

    # Define the prompt template with an introduction in Bahasa Indonesia
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful chatbot"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )

    # Combine the prompt and model into a runnable and store in the session
    runnable = (
        augment_prompt
        | RunnablePassthrough.assign(
            history=RunnableLambda(memory.load_memory_variables)
            | itemgetter("history"))
        | prompt
        | model
        | StrOutputParser())

    cl.user_session.set("memory", memory)
    cl.user_session.set("runnable", runnable)


@cl.on_message
async def on_message(message: cl.Message):
    memory = cl.user_session.get("memory")  # type: ConversationBufferMemory
    runnable = cl.user_session.get("runnable")  # Retrieve the stored runnable

    msg = cl.Message(content="")

    async for chunk in runnable.astream(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()

    memory.chat_memory.add_user_message(message.content)
    memory.chat_memory.add_ai_message(msg.content)
