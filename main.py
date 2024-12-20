# import streamlit as st
# import os
# import time
# from langchain_groq import ChatGroq
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains import create_retrieval_chain
# from langchain_chroma import Chroma
# import warnings

# warnings.filterwarnings("ignore", category=FutureWarning)

# # Load vector database
# def load_from_vectordb():
#     CHROMA_DB_PATH = 'vectorstore/db_chroma'
#     if not os.path.exists(CHROMA_DB_PATH):
#         raise FileNotFoundError(f"Could not find the VectorDB at {CHROMA_DB_PATH}")
#     embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})
#     db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
#     return db

# # Initialize the vector store and LLM
# vectors = load_from_vectordb()
# groq_api_key = "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"  # Replace with your actual Groq API key
# llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# # Streamlit interface
# st.title("Guidance & Counselling Chatbot")

# prompt = ChatPromptTemplate.from_template(
# """
# You are a knowledgeable and empathetic guidance and counselling assistant. Your role is to help high school students with concise, accurate, and relevant answers to their questions. Use the provided context and metadata to craft responses that are short and focused, ensuring they directly address the query without unnecessary details.

# Context:
# {context}

# Metadata:
# {metadata}

# Student's Question: {input}

# Guidelines for Your Response:
# 1. **Conciseness**:
#    - Keep your response short and to the point, providing only the essential information needed to answer the query.
#    - Avoid lengthy explanations or additional details unless explicitly requested.

# 2. **Personalized and Friendly**:
#    - Address the student directly in a tone that matches their query. Use a friendly tone for personal or emotional questions and a professional tone for academic or career-related guidance.

# 3. **Contextual Relevance**:
#    - Prioritize information from the provided context and metadata. If no context is available, use your general expertise or guide the student on where to find reliable information.

# 4. **Handling Missing Information**:
#    - If the context lacks relevant data, briefly inform the student and suggest where they can find accurate information.

# 5. **Empathy and Guidance**:
#    - Show empathy when answering sensitive or personal questions. Provide actionable advice where appropriate.

# 6. **Examples of Ideal Responses**:
#    - Academic Query: "The recommended books for JAMB Physical and Health Education are [Book 1, Book 2]."
#    - Personal Query: "It's okay to feel this way. Take breaks and talk to someone you trust."
#    - Missing Information: "I couldn’t find this information here. Please check the official JAMB website for the latest updates."

# Remember, your goal is to provide just enough information to effectively assist the student without overwhelming them.
# """
# )

# # Create retrieval chain
# retriever = vectors.as_retriever(search_type="similarity", search_kwargs={"k": 10})
# document_chain = create_stuff_documents_chain(llm, prompt)
# retrieval_chain = create_retrieval_chain(retriever, document_chain)

# # User prompt input
# user_prompt = st.text_input("Ask your question here:")

# if user_prompt:
#     start_time = time.process_time()
    
#     # Retrieve relevant context
#     result = retriever.invoke(user_prompt)
#     context = [doc.page_content for doc in result]
#     metadata = [doc.metadata for doc in result]
    
#     # Generate response
#     response = retrieval_chain.invoke({
#         "input": user_prompt,
#         "context": "\n\n".join(context),
#         "metadata": str(metadata)
#     })
    
#     elapsed_time = time.process_time() - start_time
    
#     # Display chatbot response
#     st.subheader("Chatbot Response:")
#     st.write(response['answer'])
#     # st.write(f"Response time: {elapsed_time:.2f} seconds")

import streamlit as st
import os
import time
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_chroma import Chroma
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# Load vector database
def load_from_vectordb():
    CHROMA_DB_PATH = 'vectorstore/db_chroma'
    if not os.path.exists(CHROMA_DB_PATH):
        raise FileNotFoundError(f"Could not find the VectorDB at {CHROMA_DB_PATH}")
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})
    db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
    return db

# Initialize the vector store and LLM
vectors = load_from_vectordb()
groq_api_key = "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"  # Replace with your actual Groq API key
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Streamlit interface
st.title("Guidance & Counselling Chatbot")

# Initialize session state
if "name" not in st.session_state:
    st.session_state["name"] = None
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Get user name if not already provided
if st.session_state["name"] is None:
    name_input = st.text_input("What is your name?", key="name_input")
    if name_input:  # Set the name only when input is provided
        st.session_state["name"] = name_input
        st.write(f"Hello {st.session_state['name']}, how can I help you today?")

if st.session_state["name"]:
    st.write(f"Hello {st.session_state['name']}, how can I help you today?")

    # Chat prompt template
    prompt = ChatPromptTemplate.from_template(
    """
    You are a knowledgeable and empathetic guidance and counselling assistant. Your role is to help high school students with concise, accurate, and relevant answers to their questions. Use the provided context and metadata to craft responses that are short and focused, ensuring they directly address the query without unnecessary details.

    Maintain a conversational flow and refer back to previous questions or answers if needed. Address the student by their name, {name}, to make the conversation personal.

    Chat History:
    {chat_history}

    Context:
    {context}

    Metadata:
    {metadata}

    Student's Question: {input}

    Guidelines for Your Response:
    1. **Conciseness**:
       - Keep your response short and to the point, providing only the essential information needed to answer the query.
       - Avoid lengthy explanations or additional details unless explicitly requested.

    2. **Personalized and Friendly**:
       - Address the student directly in a tone that matches their query. Use a friendly tone for personal or emotional questions and a professional tone for academic or career-related guidance.

    3. **Contextual Relevance**:
       - Prioritize information from the provided context and metadata. If no context is available, use your general expertise or guide the student on where to find reliable information.

    4. **Handling Missing Information**:
       - If the context lacks relevant data, briefly inform the student and suggest where they can find accurate information.

    5. **Empathy and Guidance**:
       - Show empathy when answering sensitive or personal questions. Provide actionable advice where appropriate.

    6. **Examples of Ideal Responses**:
       - Academic Query: "The recommended books for JAMB Physical and Health Education are [Book 1, Book 2]."
       - Personal Query: "It's okay to feel this way. Take breaks and talk to someone you trust."
       - Missing Information: "I couldn’t find this information here. Please check the official JAMB website for the latest updates."

    Remember, your goal is to provide just enough information (not too long and not too short) to effectively assist the student without overwhelming them.
    """
    )

    # Create retrieval chain
    retriever = vectors.as_retriever(search_type="similarity", search_kwargs={"k": 10})
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # User prompt input
    user_prompt = st.text_input("Ask your question here:")

    if user_prompt:
        start_time = time.process_time()
        
        # Retrieve relevant context
        result = retriever.invoke(user_prompt)
        context = [doc.page_content for doc in result]
        metadata = [doc.metadata for doc in result]
        
        # Generate response
        response = retrieval_chain.invoke({
            "input": user_prompt,
            "context": "\n\n".join(context),
            "metadata": str(metadata),
            "chat_history": "\n\n".join(st.session_state["chat_history"]),
            "name": st.session_state["name"]
        })
        
        elapsed_time = time.process_time() - start_time

        # Store response in chat history
        st.session_state["chat_history"].append(f"You: {user_prompt}")
        st.session_state["chat_history"].append(f"Bot: {response['answer']}")
        
        # Display chatbot response
        st.subheader("Chatbot Response:")
        st.write(response['answer'])

