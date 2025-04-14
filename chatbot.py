import os
from dotenv import load_dotenv
load_dotenv()

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["MISTRAL_API_KEY"] = os.getenv("MISTRAL_API_KEY")

from langchain.chat_models import init_chat_model

llm = init_chat_model("mistral-large-latest", model_provider="mistralai")

from langchain_mistralai import MistralAIEmbeddings

embeddings = MistralAIEmbeddings(model="mistral-embed")

from langchain_community.vectorstores import FAISS
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore

VECTOR_DB_PATH = "my_faiss_index/"
if os.path.exists(os.path.join(VECTOR_DB_PATH, "index.faiss")):
    vector_store = FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)

#Intializing FAISS vector sote to store the chunks of data  
else:        
    embedding_dim = len(embeddings.embed_query("hello world"))
    index = faiss.IndexFlatL2(embedding_dim)

    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader

    # Load a PDF file
    DATA_PATH="C:/Users/vijay/Documents/Programming/chatbot/ChemistryPDF"
    loader=DirectoryLoader(path=DATA_PATH,glob='*.pdf',loader_cls=PyPDFLoader)
    docs = loader.load()

    from langchain_text_splitters import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # chunk size (characters)
        chunk_overlap=200,  # chunk overlap (characters)
        add_start_index=True,  # track index in original document
    )
    all_splits = text_splitter.split_documents(docs)

    print(f"Split blog post into {len(all_splits)} sub-documents.")

    document_ids = vector_store.add_documents(documents=all_splits)
    vector_store.save_local("my_faiss_index/")

    vector_store = FAISS.load_local("my_faiss_index/", embeddings, allow_dangerous_deserialization=True)

from langchain import hub
from langgraph.graph import START, StateGraph
from langchain_core.documents import Document
from typing_extensions import List, TypedDict

# Define prompt for question-answering
# prompt = hub.pull("rlm/rag-prompt") predifined prompt template

from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an expert Organic Chemistry assistant. Use only the provided context as your primary source of information. 
If the context contains enough information, answer strictly based on it. 
If the context provides only partial information, build upon it using your own organic chemistry knowledge — but stay in the domain of organic chemistry. 

Do NOT answer questions that are completely irrelevant to organic chemistry or when the context provides no clues. 
If you're unsure or the context is totally unrelated to the question, reply with: "I don't have enough information to answer that."

Context:
{context}

Question:
{question}

Answer:
"""
)


# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# Define application steps
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"],k=10)
    return {"context": retrieved_docs}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

# Compile application and test
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

def ask_question(input):
    try:
        question=input
        # Add question to state
        state_input = {"question": question,}

        # Run RAG
        state_output = graph.invoke(state_input)
            
        answer = state_output["answer"]

        return answer
    except Exception as e:
        print("Error in bot as ",e)
        return False