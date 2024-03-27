import streamlit as st
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate 


os.environ["GOOGLE_API_KEY"] = str("AIzaSyB7OZmQj3LQ2SiVPg6dBPuqsWk51x3i74c")
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))


def get_conversational_chain():  

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details. 

    don't give wrong answers or made up answers by hallucination.
    
    \n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0, max_tokens=300,request_timeout=3)
    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    new_db = FAISS.load_local("faiss_prod_inex", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)
    #print(response)
    st.write("Reply: ", response["output_text"])


def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini")
    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)



if __name__ == "__main__":
    main()