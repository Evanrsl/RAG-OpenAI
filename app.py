import openai
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



# def augment_prompt(prompt: str):
#     # get top 3 results from knowledge base
#     results = index.query(
#         vector=embedding(prompt['question']),
#         top_k=5,
#         include_metadata=True
#     )

#     source_knowledge = ""
#     num_of_context = 0
#     for result in results['matches']:
#         score = result['score']
#         text = result['metadata']['text']
#         print(score)
#         if score  > 0.85 and len(text) > 50:
#             print(text)
#             source_knowledge += f"text: {text} \n source:{result['metadata']['source']} \n\n"
#             num_of_context += 1
#     print(num_of_context)
#     # feed into an augmented prompt
#     if num_of_context > 0 :
#         augmented_prompt = f"""{prompt['question']}

#         gunakan informasi ini jika berguna:
#         {source_knowledge}"""
#         print(augmented_prompt)
#         return {"question": f"{augmented_prompt}"}
#     return prompt


# Chainlit

@cl.on_chat_start
async def on_chat_start():
    memory = ConversationBufferMemory(return_messages=True)

    # Initialize the model with specific configurations
    model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)



    # Contextualizing the question

    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )
    contextualize_q_chain = contextualize_q_prompt | model | StrOutputParser()


    # Chain with chat history

    qa_system_prompt = """You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \
    Use three sentences maximum and keep the answer concise.\

    {context}"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    def contextualized_question(input: dict):
        if input.get("chat_history"):
            return contextualize_q_chain
        else:
            return input["question"]


    rag_chain = (
        RunnablePassthrough.assign(
            context=contextualized_question | retriever | format_docs
        )
        | qa_prompt
        | model
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
