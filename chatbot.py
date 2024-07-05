

pip install streamlit pypdf2 langchain faiss-cpu



import streamlit as st

from PyPDF2 import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter

pip install -U langchain-community

from langchain.embeddings.openai import OpenAIEmbeddings

from langchain.vectorstores import FAISS

from langchain.chains.question_answering import load_qa_chain

from langchain_community.chat_models import ChatOpenAI

OPENAI_API_KEY="Paste here your api key genrate by openai api"

st.header("CHATBOT")

with st.sidebar:
  st.title("Documents ")
  file=st.file_uploader("Upload your Pdf File here ",type="pdf")

if file is not None:
  pdf_reader=PdfReader(file)
  text =""
  for page in pdf_reader.pages:
    text +=page.extract_text()
    st.write(text)

#Break into chunks
    text_splitter=RecursiveCharacterTextSplitter(
       separators="\n",
       chunk_size=1000,
       chunk_overlap=150,
       lenght_function=len
    )

    chunks =text_splitter.split_text(text)

    #generate the embeddings

    embeddings=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    #creating the vector-store-Faiss

    vector_store=FAISS.from_texts(chunks,embeddings)

    #get the user questions
    user_question=st.text_input("Type the questions here")


    #do similarity checks

    if user_question:
      match=vector_store.smilarity_search(user_question)




      llm =ChatOpenAI(
          open_api_key= OPENAI_API_KEY,
          temperature=0,
          max_tokens=1000,
          model_name="gpt-3.5-turbo"

      )
      #output the results
      #chain -->take the question,get relevent document ,pass it to llm,genrerate it


      chain =load_qa_chain(llm,chain_type="stuff")
      response=chain.run(input_documents=match,question= user_question)
      st.write(response)