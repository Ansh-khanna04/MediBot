import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from euriai import EuriaiClient
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from config import (
    EURI_API_KEY, EURI_BASE_URL, EMBEDDING_MODEL_NAME,
    VECTOR_STORE_PATH, CHAT_MODEL_NAME
)

# In-memory chat history
chat_history = []

# Call Euri directly — no ChatOpenAI
def call_euri(prompt_value):
    client = EuriaiClient(api_key=EURI_API_KEY,
                          model= CHAT_MODEL_NAME)
    messages = prompt_value.to_messages()
    # Convert to plain text for Euri
    full_prompt = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
        for m in messages
    )
    response = client.generate_completion(
        prompt=full_prompt,
        max_tokens=1024,
        temperature=0.6
    )
    return response["choices"][0]["message"]["content"]


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

def load_chain():
    embeddings = get_embeddings()

    if not os.path.exists(VECTOR_STORE_PATH):
        raise FileNotFoundError(
            f"Vector store not found at '{VECTOR_STORE_PATH}'. Run ingest.py first."
        )

    vector_store = FAISS.load_local(
        VECTOR_STORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    #llm = get_chat_model(CHAT_MODEL_NAME, 0.6, EURI_API_KEY)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are MediBot, a helpful medical document assistant.
Answer ONLY using the context provided below.
Always mention the source filename and page number when referencing information.
If the answer is not found in the context, say:
'I don't have enough information in the provided documents to answer this.'
Do NOT make up any medical information.

Context:
{context}"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])

    def format_docs(docs):
        return "\n\n".join(
            f"[Source: {doc.metadata.get('source', 'unknown')}, "
            f"Page: {doc.metadata.get('page', '?')}]\n{doc.page_content}"
            for doc in docs
        )

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
            "chat_history": lambda _: chat_history
        }
        | prompt
        | RunnableLambda(call_euri)
        | StrOutputParser()
    )

    return chain, retriever