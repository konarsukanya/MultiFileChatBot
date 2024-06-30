import os
import logging
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from a .env file if present
# from dotenv import load_dotenv
# load_dotenv()

# Get environment variables
azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_openai_key = os.getenv("AZURE_OPENAI_API_KEY")

logger.info(f"Azure OpenAI Endpoint: {azure_openai_endpoint}")

# Set the environment variables in the script if not set already
os.environ["AZURE_OPENAI_ENDPOINT"] = azure_openai_endpoint
os.environ["AZURE_OPENAI_API_KEY"] = azure_openai_key

# Verify environment variables are loaded
if not azure_openai_endpoint or not azure_openai_key:
    raise ValueError("Environment variables for Azure OpenAI endpoint or key are not set.")

# Step 1: Load documents from a directory and combine their text
document_directory = "pdf_docs"
if not os.path.exists(document_directory):
    raise FileNotFoundError(f"The directory {document_directory} does not exist.")

# Collect all PDF files in the directory
pdf_files = [os.path.join(document_directory, f) for f in os.listdir(document_directory) if f.endswith('.pdf')]
if not pdf_files:
    raise FileNotFoundError(f"No PDF files found in the directory {document_directory}.")

combined_text = ""
for file_path in pdf_files:
    loader = PyPDFLoader(file_path)
    loaded_documents = loader.load()
    for doc in loaded_documents:
        combined_text += doc.page_content + "\n"  # Combine all text from all documents
        logger.info(f"Loaded text from {file_path}")

# Check the combined text length
logger.info(f"Combined text length: {len(combined_text)} characters")

# Step 2: Create a LangChain Document object for the combined text
document = Document(page_content=combined_text)

# Step 3: Create embeddings using Azure OpenAI
embeddings = AzureOpenAIEmbeddings(
    azure_deployment="text-embedding-ada-002",
    openai_api_version="2023-03-15-preview"
)

# Step 4: Create a vector store from the document
vector_store = FAISS.from_documents([document], embeddings)

# Step 5: Create a conversational retrieval chain with a custom prompt
prompt_template = """
You are an AI-driven bot. Your task is to answer queries based on the provided documents. Your responses should strictly adhere to the information presented in the documents, providing a seamless flow of information from the document to the user. Your responses must be Spartan—brief and direct. Do not engage with abusive queries.
Please provide concise yet comprehensive answers.
Don't justify your answers. Don't give information not mentioned in the CONTEXT INFORMATION. If the answer is not in the context, say the words "Sorry, I am unable to answer your question with the information available to me"

Query: {question}

Context:
{context}

Answer:
"""
prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

llm = AzureChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.1,
    azure_deployment='gpt-35-turbo',
    openai_api_version="2023-03-15-preview"
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(),
    verbose=True,
    return_source_documents=True,
    chain_type_kwargs={
        "prompt": prompt
    }
)


while True:
    query = input("Please enter your query (type 'exit' to quit): ").strip()
    if query.lower() == 'exit':
        print("Thank you for using the bot!")
        break
    
    try:
        response = qa_chain({
            "query": query
        })
        answer = response["result"]
        print(answer)
    except Exception as e:
        logger.error(f"Could not complete the querying process. Error: {e}")
        print("Some issue occurred, please try again")
