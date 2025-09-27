import json
import os
import sys
import boto3
import streamlit as st
## We will be suing Titan Embeddings Model To generate Embedding

from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
## Data Ingestion

import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
# Vector Embedding And Vector Store

from langchain.vectorstores import FAISS

## LLm Models
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
## Bedrock Clients
bedrock=boto3.client(service_name="bedrock-runtime")
## call the embedding from the aws bedrock
bedrock_embeddings=BedrockEmbeddings(model_id="amazon.titan-embed-text-v1",client=bedrock)

# Page setup
st.set_page_config("Chat with PDF")


## Data ingestion
def data_ingestion():
    print("Loading PDF documents from 'data' folder...")
    if not os.path.exists("data"):
        st.error("The 'data' folder does not exist. Please create it and add PDF files.")
        return []
    loader=PyPDFDirectoryLoader("data")
    documents=loader.load()
    print(f"Loaded {len(documents)} documents.")
    if not documents:
        st.error("No PDFs found in 'data' folder.")
        return []
    # - in our testing Character split works better with this PDF data set
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000,
                                                 chunk_overlap=1000)
    
    docs=text_splitter.split_documents(documents)
    print(f"Split documents into {len(docs)} chunks.")
    return docs

## Vector Embedding and vector store

def get_vector_store(docs):
    print("GET from FAISS vector store..."+str(docs))
    if not docs:
        st.error("No documents to embed.")
        return

    try:
        print("Trying to print"+str(docs))
        print("Generating embeddings and creating FAISS vector store..."+docs[0].page_content[:100])
        vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
        print("Creating FAISS vector store...")
        vectorstore_faiss.save_local("faiss_index")
        print("FAISS index saved successfully!")

    except Exception as e:
        st.error(f"Failed to create vector store: {e}")



def get_claude_llm():
    ##create the Anthropic Model
    llm=Bedrock(model_id="ai21.j2-mid-v1",client=bedrock,
                model_kwargs={'maxTokens':512})
    
    return llm


def get_llama2_llm():
    ##create the Anthropic Model
    llm=Bedrock(model_id="meta.llama3-8b-instruct-v1:0",client=bedrock,
                model_kwargs={'max_gen_len':512})
    
    return llm  


prompt_template = """

Human: Use the following pieces of context to provide a 
concise answer to the question at the end but use atleast summarize with 
250 words with detailed explanations. If you don't know the answer, 
just say that you don't know, don't try to make up an answer.
<context>
{context}
</context>

Question: {question}

Assistant:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
def get_response_llm(llm,vectorstore_faiss,query):
    qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore_faiss.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    ),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)
    answer=qa({"query":query})
    return answer['result']

def main():
    st.set_page_config("Chat PDF")
   

    st.title("Bedrock LLM with FAISS Vector Store")
    st.sidebar.title("Configuration")
    llm_option = st.sidebar.selectbox(
        "Select LLM Model",
        ("Claude (AI21)", "Llama2 (Meta)"),
    )

    if llm_option == "Claude (AI21)":
        llm = get_claude_llm()
    else:
        llm = get_llama2_llm()

    if os.path.exists("faiss_index"):
      print("Loading existing FAISS index...")
      vectorstore_faiss = FAISS.load_local(
        "faiss_index",
        bedrock_embeddings,
        allow_dangerous_deserialization=True
      )
    else:
      docs = data_ingestion()
      get_vector_store(docs)
      vectorstore_faiss = FAISS.load_local(
        "faiss_index",
        bedrock_embeddings,
        allow_dangerous_deserialization=True
      )


    query = st.text_input("Enter your question:")
    if st.button("Get Answer"):
        if query:
            with st.spinner("Generating answer..."):
                answer = get_response_llm(llm, vectorstore_faiss, query)
            st.subheader("Answer:")
            st.write(answer)
        else:
            st.warning("Please enter a question."
            )
if __name__ == "__main__":
    main()

