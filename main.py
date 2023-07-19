import os
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from config import oak
from pydantic import BaseModel

# LangChain PDF Data Loaders
os.environ['OPENAI_API_KEY'] = oak

loader = PyMuPDFLoader("./docs/H2O_-_Augment_Hack_PDF.pdf")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=10)
texts = text_splitter.split_documents(documents)

# Leveraging embeddings
persist_directory = "./storage"
embeddings = OpenAIEmbeddings()
vectordb = Chroma.from_documents(documents=texts, 
                                 embedding=embeddings,
                                 persist_directory=persist_directory)
vectordb.persist()

# Chatting with PDF Documents
retriever = vectordb.as_retriever()
llm = ChatOpenAI(model_name='gpt-3.5-turbo-16k-0613')
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

def chatbot(user_input, qa):
    query = f"###Prompt {user_input}"
    try:
        llm_response = qa(query)
        result = llm_response["result"]
        print(result)
    except Exception as err:
        print('Exception occurred. Please try again', str(err))
    return result

app = FastAPI()

origins = [
    "http://localhost:3000",
    "https://tokengated-next-chatgpt-ochre.vercel.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
    
class Message(BaseModel):
    role: str
    content: str

@app.post("/get_last_user_message/")
async def get_last_user_message(payload: dict):
    messages = payload.get("messages", [])
    
    # Filter the messages by "user" role and get the last one
    user_messages = [message["content"] for message in messages if message["role"] == "user"]
    
    if user_messages:
        return chatbot(user_messages[-1], qa)
    else:
        return 'No message from user.'

if __name__ == "__main__":
    import uvicorn

    app.add_middleware(HTTPSRedirectMiddleware)
    uvicorn.run(app, host='157.245.65.240', port=8000,
                ssl_keyfile="/etc/letsencrypt/live/api.kssoftware.xyz/privkey.pem",
                ssl_certfile="/etc/letsencrypt/live/api.kssoftware.xyz/fullchain.pem")