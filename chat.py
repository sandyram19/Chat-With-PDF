import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain.chains import ConversationChain
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts.prompt import PromptTemplate
from langchain.document_loaders import PyPDFLoader 
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")


FILE_PATH="Corpus.pdf"


def get_pdf_text():
    text=""
    pdf_reader= PdfReader(FILE_PATH)
    for page in pdf_reader.pages:
        text+= page.extract_text()
    return  text



def get_text_chunks(doc):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(doc)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(google_api_key=GOOGLE_API_KEY,model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

if "pdf" not in st.session_state:
    raw_text = get_pdf_text()
    text_chunks = get_text_chunks(raw_text)
    # st.write(text_chunks)
    get_vector_store(text_chunks)
    st.session_state["pdf"] = True






def user_input(user_question,chain):
    embeddings = GoogleGenerativeAIEmbeddings(google_api_key=GOOGLE_API_KEY, model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    print("\nDOCS\n",docs)
    history = chain.memory.buffer
    print("HIstory\n",history)
    response=chain({"input_documents": docs, "human_input": user_question}, return_only_outputs=True)
    return response["output_text"]
    # st.write("Reply: ", response["output_text"])




def main():
    st.set_page_config(page_title="Santhosh Ram G K", page_icon=":medal:", layout="wide")
    st.header("Jessup Cellars chatbot :robot_face: ")
    
    if "conversation_chain" not in st.session_state:
        
        prompt_template = """
        Answer the question as detailed as possible from the provided context, make sure to provide all the details. If questions are asked outside the context, reply "Contact business directly."
        
        Context:\n {context}\n

        {chat_history}\n
        Human: {human_input}\n

        Answer:
        """

        prompt = PromptTemplate(input_variables = ["chat_history","human_input", "context"],template = prompt_template)
        memory=ConversationBufferMemory(memory_key="chat_history",input_key="human_input")
        llm = ChatGoogleGenerativeAI(google_api_key=GOOGLE_API_KEY, model="gemini-pro",
                                temperature=0.3)

        
        chain =  load_qa_chain(llm=llm,  memory=memory, prompt=prompt, chain_type="stuff")
        

        st.session_state["conversation_chain"] = chain
    conversation=st.session_state["conversation_chain"]

    if "messages" not in st.session_state:
        st.session_state.messages = []


    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
        
    # React to user input
    if user_question := st.chat_input("What is up?"):
        res=user_input(user_question,conversation)
        # Display user message in chat message container
        st.chat_message("user").markdown(user_question)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_question})

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(res)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": res})

if __name__ == "__main__":
    main()
