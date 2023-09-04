import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
def get_raw_text(pdf_docs):
    text = ''
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for pages in pdf_reader.pages:
            text += pages.extract_text()
    return text

def get_text_chunks(text):
    text_splliter = CharacterTextSplitter(
        separator='\n',
        chunk_size=50,
        chunk_overlap=5,
        length_function=len
    )
    chunks = text_splliter.split_text(text)
    return chunks

def get_vectorestore(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorestore = FAISS.from_texts(texts=text_chunks, embeddings=embeddings)
    return vectorestore

def get_conversation(vectorstore):
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        memory=memory,
        retriever=vectorstore.as_retriever()
    )
    return conversation_chain

def handle_user_input(user_question):
    response = st.session_state.conversation({'question':user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i%2==0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title='Chat with your PDFs', page_icon=":books:")

    st.write(css, unsafe_allow_html=True)

    if 'conversation' not in st.session_state:
        st.session_state.conversation=None

    st.header('Chat with your PDFs :books:')
    user_question = st.text_input('Enter your query here')

    if user_question:
        handle_user_input(user_question)

    with st.sidebar:
        st.subheader('Your Documents')
        pdf_docs = st.file_uploader('Upload your files here and press "Process"', accept_multiple_files=True)
        if st.button('Process'):
            with st.spinner('Processing'):

                #raw text from all pdfs
                raw_text = get_raw_text(pdf_docs)

                #breaking the text into text chunks
                text_chunks = get_text_chunks(raw_text)

                #converting the chunks into embeddings and storing them in vector database
                vectorstore = get_vectorestore(text_chunks)

                #creating conversational chatbot
                st.session_state.conversation = get_conversation(vectorstore)



if __name__ == '__main__':
    main()