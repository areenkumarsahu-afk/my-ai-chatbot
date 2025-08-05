import streamlit as st 
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
import tempfile
import os 

def load_docs(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path=tmp_file.name
    with open(file_path,'r',encoding='utf-8') as f:
        raw_text=f.read()
    text_splitter=CharacterTextSplitter(separator='/n',chunk_size=1000,chunk_overlap=200)
    docs=text_splitter.create_documents([raw_text])
    return docs

def get_answers(docs,query):
    embeddings=OpenAIEmbeddings()
    vectorstore=FAISS.from_documents(docs,embeddings)
    retriever=vectorstore.as_retriever()
    relevant_docs=retriever.get_relevent_documents(query)
    llm=ChatOpenAI(temprature=0)
    chain=load_qa_chain(llm,chain_type='stuff')
    answer=chain.run(input_documents=relevant_docs,question=query)
    return answer

st.title("AI Chatbot with Your Documents")
uploaded_file=st.file_uploader("Upload a file")
if uploaded_file:
    docs=load_docs(uploaded_file)
    query=st.text_input("Ask something:")
    if query:
        answer=get_answers(docs,query)

        st.write("Answer:",answer)
