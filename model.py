import inspect
import json
import logging
import os
import re
import shutil
from tempfile import NamedTemporaryFile
from typing import Dict, List

import gradio as gr
import requests
from duckduckgo_search import DDGS
from huggingface_hub import InferenceClient
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from pydantic import BaseModel, Field

from llama_parse import LlamaParse

# --------------------------------------------------------------------------------
# Logging Configuration
# --------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Starting application...")

# --------------------------------------------------------------------------------
# Environment Variables and Global Constants
# --------------------------------------------------------------------------------
HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")
LLAMA_CLOUD_API_KEY = os.environ.get("LLAMA_CLOUD_API_KEY")
ACCOUNT_ID = os.environ.get("CLOUDFARE_ACCOUNT_ID")
API_TOKEN = os.environ.get("CLOUDFLARE_AUTH_TOKEN")
API_BASE_URL = "https://api.cloudflare.com/client/v4/accounts/a17f03e0f049ccae0c15cdcf3b9737ce/ai/run/"
DOCUMENTS_FILE = "uploaded_documents.json"

MODELS = [
    "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "@cf/meta/llama-3.1-8b-instruct",
    "mistralai/Mistral-Nemo-Instruct-2407",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "duckduckgo/gpt-4o-mini",
    "duckduckgo/claude-3-haiku",
    "duckduckgo/llama-3.1-70b",
    "duckduckgo/mixtral-8x7b"
]

logging.info(f"ACCOUNT_ID: {ACCOUNT_ID}")
if API_TOKEN:
    logging.info(f"CLOUDFLARE_AUTH_TOKEN: {API_TOKEN[:5]}...")
else:
    logging.info("CLOUDFLARE_AUTH_TOKEN not set.")

# --------------------------------------------------------------------------------
# Global Variables
# --------------------------------------------------------------------------------
uploaded_documents = []

# --------------------------------------------------------------------------------
# Initialize LlamaParse
# --------------------------------------------------------------------------------
llama_parser = LlamaParse(
    api_key=LLAMA_CLOUD_API_KEY,
    result_type="markdown",
    num_workers=4,
    verbose=True,
    language="en",
)

# --------------------------------------------------------------------------------
# Helper Functions
# --------------------------------------------------------------------------------

def load_documents_from_json() -> List[Dict]:
    """
    Load previously uploaded documents from a local JSON file.
    Returns a list of dicts representing documents.
    """
    if os.path.exists(DOCUMENTS_FILE):
        with open(DOCUMENTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_documents_to_json(documents: List[Dict]) -> None:
    """
    Save the current list of uploaded documents to a local JSON file.
    """
    with open(DOCUMENTS_FILE, "w", encoding="utf-8") as f:
        json.dump(documents, f)

def display_documents() -> gr.CheckboxGroup:
    """
    Returns a Gradio CheckboxGroup for the user to select uploaded documents.
    """
    return gr.CheckboxGroup(
        choices=[doc["name"] for doc in uploaded_documents],
        value=[doc["name"] for doc in uploaded_documents if doc["selected"]],
        label="Select documents to query or delete"
    )

def load_document(file: NamedTemporaryFile, parser: str = "llamaparse") -> List[Document]:
    """
    Load and split the given PDF file into Document chunks.
    Supports two parsers: 'pypdf' or 'llamaparse'.
    """
    if parser == "pypdf":
        loader = PyPDFLoader(file.name)
        return loader.load_and_split()
    elif parser == "llamaparse":
        try:
            documents = llama_parser.load_data(file.name)
            return [
                Document(page_content=doc.text, metadata={"source": file.name})
                for doc in documents
            ]
        except Exception as e:
            logging.error(f"Error using LlamaParse: {str(e)}. Falling back to PyPDF parser.")
            loader = PyPDFLoader(file.name)
            return loader.load_and_split()
    else:
        raise ValueError("Invalid parser specified. Use 'pypdf' or 'llamaparse'.")

def get_embeddings():
    """
    Return a HuggingFace embeddings model.
    """
    return HuggingFaceEmbeddings(model_name="avsolatorio/GIST-Embedding-v0")

# --------------------------------------------------------------------------------
# Document Management
# --------------------------------------------------------------------------------

def update_vectors(files, parser):
    """
    Load PDF files, parse them into chunks, and store them in a local FAISS index.
    Also updates the global 'uploaded_documents' list.
    """
    global uploaded_documents
    logging.info(f"Entering update_vectors with {len(files)} files and parser: {parser}")
    
    if not files:
        logging.warning("No files provided for update_vectors.")
        return "Please upload at least one PDF file.", display_documents()
    
    embed = get_embeddings()
    total_chunks = 0
    all_data = []
    
    for file in files:
        logging.info(f"Processing file: {file.name}")
        try:
            data = load_document(file, parser)
            if not data:
                logging.warning(f"No chunks loaded from {file.name}")
                continue
            logging.info(f"Loaded {len(data)} chunks from {file.name}")
            all_data.extend(data)
            total_chunks += len(data)

            # Track uploaded documents
            if not any(doc["name"] == file.name for doc in uploaded_documents):
                uploaded_documents.append({"name": file.name, "selected": True})
                logging.info(f"Added new document to 'uploaded_documents': {file.name}")
            else:
                logging.info(f"Document already exists: {file.name}")
        except Exception as e:
            logging.error(f"Error processing file {file.name}: {str(e)}")
    
    logging.info(f"Total chunks processed: {total_chunks}")
    
    if not all_data:
        logging.warning("No valid data extracted from uploaded files.")
        return (
            "No valid data could be extracted. Check file contents and try again.",
            display_documents()
        )
    
    # Update or create a FAISS database
    try:
        if os.path.exists("faiss_database"):
            logging.info("Updating existing FAISS database.")
            database = FAISS.load_local("faiss_database", embed, allow_dangerous_deserialization=True)
            database.add_documents(all_data)
        else:
            logging.info("Creating new FAISS database.")
            database = FAISS.from_documents(all_data, embed)
        
        database.save_local("faiss_database")
        logging.info("FAISS database saved.")
    except Exception as e:
        logging.error(f"Error updating FAISS database: {str(e)}")
        return f"Error updating vector store: {str(e)}", display_documents()

    # Save updated documents list
    save_documents_to_json(uploaded_documents)

    return (
        f"Vector store updated successfully. Processed {total_chunks} chunks "
        f"from {len(files)} files using {parser}.",
        display_documents()
    )

def delete_documents(selected_docs):
    """
    Delete documents from FAISS index and update global 'uploaded_documents' accordingly.
    If all documents are deleted, remove the FAISS directory entirely.
    """
    global uploaded_documents
    
    if not selected_docs:
        return "No documents selected for deletion.", display_documents()
    
    if not os.path.exists("faiss_database"):
        logging.warning("FAISS database not found for deletion.")
        return "No FAISS database found.", display_documents()
    
    embed = get_embeddings()
    database = FAISS.load_local("faiss_database", embed, allow_dangerous_deserialization=True)
    
    deleted_docs = []
    docs_to_keep = []
    # Filter out documents to keep vs. delete
    for doc in database.docstore._dict.values():
        if doc.metadata.get("source") not in selected_docs:
            docs_to_keep.append(doc)
        else:
            deleted_docs.append(doc.metadata.get("source", "Unknown"))
    
    logging.info(f"Total documents before deletion: {len(database.docstore._dict)}")
    logging.info(f"Keeping {len(docs_to_keep)} documents; deleting {len(deleted_docs)}.")
    
    if not docs_to_keep:
        # If all documents are deleted, remove the FAISS database directory
        if os.path.exists("faiss_database"):
            shutil.rmtree("faiss_database")
        logging.info("All documents deleted. Removed FAISS database.")
    else:
        # Create new FAISS index with remaining documents
        new_database = FAISS.from_documents(docs_to_keep, embed)
        new_database.save_local("faiss_database")
        logging.info(f"Created new FAISS index with {len(docs_to_keep)} documents.")
    
    # Update the global 'uploaded_documents'
    uploaded_documents = [doc for doc in uploaded_documents if doc["name"] not in deleted_docs]
    save_documents_to_json(uploaded_documents)
    
    return f"Deleted documents: {', '.join(deleted_docs)}", display_documents()

# --------------------------------------------------------------------------------
# Text Generation & Chat Logic
# --------------------------------------------------------------------------------

def generate_chunked_response(
    prompt,
    model,
    max_tokens=10000,
    num_calls=3,
    temperature=0.2,
    should_stop=False
):
    """
    Generates a response in chunks from either Cloudflare's Llama-based API
    or Hugging Face's InferenceClient (depending on the 'model' argument).
    """
    logging.info(f"Starting generate_chunked_response with model={model}, num_calls={num_calls}")
    full_response = ""
    
    # Prepare messages for conversation
    messages = [{"role": "user", "content": prompt}]
    
    if model == "@cf/meta/llama-3.1-8b-instruct":
        # Cloudflare API logic
        for i in range(num_calls):
            if should_stop:
                logging.info("Stop triggered, aborting Cloudflare calls.")
                break
            logging.info(f"Cloudflare API call {i+1}")
            try:
                response = requests.post(
                    f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/run/{model}",
                    headers={"Authorization": f"Bearer {API_TOKEN}"},
                    json={
                        "stream": True,
                        "messages": [
                            {"role": "system", "content": "You are a friendly assistant"},
                            {"role": "user", "content": prompt}
                        ],
                        "max_tokens": max_tokens,
                        "temperature": temperature
                    },
                    stream=True
                )
                
                for line in response.iter_lines():
                    if should_stop:
                        logging.info("Stop triggered during streaming.")
                        break
                    if line:
                        try:
                            json_data = json.loads(line.decode('utf-8').split('data: ')[1])
                            chunk = json_data['response']
                            full_response += chunk
                        except json.JSONDecodeError:
                            continue
                logging.info(f"Cloudflare API call {i+1} completed.")
            except Exception as e:
                logging.error(f"Error in Cloudflare response: {str(e)}")
    else:
        # Hugging Face API logic
        client = InferenceClient(model, token=HUGGINGFACE_TOKEN)
        for i in range(num_calls):
            if should_stop:
                logging.info("Stop triggered, aborting HF calls.")
                break
            logging.info(f"Hugging Face API call {i+1}")
            try:
                for message in client.chat_completion(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=True,
                ):
                    if should_stop:
                        logging.info("Stop triggered during streaming.")
                        break
                    if (message.choices and 
                        message.choices[0].delta and 
                        message.choices[0].delta.content):
                        chunk = message.choices[0].delta.content
                        full_response += chunk
                logging.info(f"Hugging Face API call {i+1} completed.")
            except Exception as e:
                logging.error(f"Error in Hugging Face response: {str(e)}")
    
    # Clean up the response
    clean_response = re.sub(r'<s>\\[INST\\].*?\\[/INST\\]\\s*', '', full_response, flags=re.DOTALL)
    clean_response = clean_response.replace("Using the following context:", "").strip()
    clean_response = clean_response.replace(
        "Using the following context from the PDF documents:", ""
    ).strip()
    
    # Remove duplicate paragraphs and sentences
    paragraphs = clean_response.split('\n\n')
    unique_paragraphs = []
    for paragraph in paragraphs:
        if paragraph not in unique_paragraphs:
            sentences = paragraph.split('. ')
            unique_sentences = []
            for sentence in sentences:
                if sentence not in unique_sentences:
                    unique_sentences.append(sentence)
            unique_paragraphs.append('. '.join(unique_sentences))
    
    final_response = '\n\n'.join(unique_paragraphs)
    logging.info(f"Final cleaned response (first 100 chars): {final_response[:100]}...")
    return final_response

def chatbot_interface(message, history, model, temperature, num_calls):
    """
    Main chatbot interface function used by Gradio.
    Handles new messages and yields updated history with responses.
    """
    if not message.strip():
        return "", history
    history = history + [(message, "")]

    try:
        for response in respond(message, history, model, temperature, num_calls):
            history[-1] = (message, response)
            yield history
    except gr.CancelledError:
        yield history
    except Exception as e:
        logging.error(f"Unexpected error in chatbot_interface: {str(e)}")
        history[-1] = (message, f"An unexpected error occurred: {str(e)}")
        yield history

def retry_last_response(history, model, temperature, num_calls):
    """
    Retry the last user query by removing the last response from history
    and calling 'chatbot_interface' again.
    """
    if not history:
        return history
    
    last_user_msg = history[-1][0]
    history = history[:-1]  # Remove the last response
    return chatbot_interface(last_user_msg, history, model, temperature, num_calls)

def truncate_context(context: str, max_length: int = 16000) -> str:
    """
    Truncate the context to a maximum character length.
    """
    if len(context) <= max_length:
        return context
    return context[:max_length] + "..."

def get_response_from_duckduckgo(query, model, context, num_calls=1, temperature=0.2):
    """
    Retrieve responses from DuckDuckGo's LLM using the specified model,
    incorporating the given context.
    """
    logging.info(f"Using DuckDuckGo chat with model: {model}")
    ddg_model = model.split('/')[-1]  # Extract the model name from the string
    truncated_context = truncate_context(context)

    full_response = ""
    for _ in range(num_calls):
        try:
            contextualized_query = f"Using the following context:\n{truncated_context}\n\nUser question: {query}"
            results = DDGS().chat(contextualized_query, model=ddg_model)
            full_response += results + "\n"
            logging.info(f"DuckDuckGo API response received. Length: {len(results)}")
        except Exception as e:
            logging.error(f"Error in DuckDuckGo: {str(e)}")
            yield f"An error occurred with the {model} model: {str(e)}. Please try again."
            return

    yield full_response.strip()

# --------------------------------------------------------------------------------
# Conversation Management
# --------------------------------------------------------------------------------
class ConversationManager:
    """
    Manages the conversation context between user queries to handle
    whether a new query is related to the previous conversation or not.
    """
    def __init__(self):
        self.history = []
        self.current_context = None

    def add_interaction(self, query, response):
        self.history.append((query, response))
        preview = response[:200]
        self.current_context = f"Previous query: {query}\nPrevious response summary: {preview}..."

    def get_context(self):
        return self.current_context

conversation_manager = ConversationManager()

# --------------------------------------------------------------------------------
# Web Search & Summaries
# --------------------------------------------------------------------------------

def get_web_search_results(query: str, max_results: int = 10) -> List[Dict[str, str]]:
    """
    Perform a DuckDuckGo text search for the given query and return up to max_results.
    """
    try:
        results = list(DDGS().text(query, max_results=max_results))
        if not results:
            logging.info(f"No results found for query: {query}")
        return results
    except Exception as e:
        logging.error(f"Error during web search: {str(e)}")
        return [{"error": f"An error occurred: {str(e)}"}]

def rephrase_query(original_query: str, conv_manager: ConversationManager) -> str:
    """
    Use the conversation context to decide if we need to rephrase the query.
    If it's a continuation of the last conversation, incorporate context.
    If not, just rephrase for clarity.
    """
    context = conv_manager.get_context()
    if context:
        prompt = f"""You are a highly intelligent conversational chatbot. Your task is to:
1. Determine if the new query is a continuation of previous conversation or a new topic.
2. If continuation, rephrase the query by incorporating relevant context.
3. If new topic, rephrase for clarity without using the previous context.
4. Provide ONLY the rephrased query without further explanation.

Context: {context}
New query: {original_query}
Rephrased query:"""
        response = DDGS().chat(prompt, model="llama-3.1-70b")
        rephrased_query = response.split('\n')[0].strip()
        return rephrased_query
    return original_query

def summarize_web_results(
    query: str,
    search_results: List[Dict[str, str]],
    conv_manager: ConversationManager
) -> str:
    """
    Summarize the given search results in a news/article style,
    incorporating the conversation context if available.
    """
    try:
        context = conv_manager.get_context()
        search_context = "\n\n".join([
            f"Title: {res['title']}\nContent: {res['body']}"
            for res in search_results if 'body' in res
        ])

        prompt = f"""You are an expert analyst. Summarize these search results about '{query}' 
while considering the context: {context}. 
Create a cohesive summary focusing on relevant information, 
citing sources inline with URLs, if any.

{search_context}

Article:"""

        summary = DDGS().chat(prompt, model="llama-3.1-70b")
        return summary
    except Exception as e:
        logging.error(f"Error in summarizing web results: {str(e)}")
        return f"An error occurred during summarization: {str(e)}"

# --------------------------------------------------------------------------------
# Core Respond Function
# --------------------------------------------------------------------------------

def respond(message, history, model, temperature, num_calls, use_web_search, selected_docs):
    """
    Dispatch function deciding whether to use web search or PDF-based search,
    then yields text responses in chunks (for streaming in Gradio).
    """
    logging.info(f"User Query: {message}")
    logging.info(f"Model: {model}, Selected Docs: {selected_docs}, Web Search: {use_web_search}")

    if use_web_search:
        original_query = message
        rephrased_query = rephrase_query(original_query, conversation_manager)
        logging.info(f"Rephrased Query: {rephrased_query}")

        final_summary = ""
        for _ in range(num_calls):
            search_results = get_web_search_results(rephrased_query)
            if not search_results:
                final_summary += f"No results found for: {rephrased_query}\n\n"
            elif "error" in search_results[0]:
                final_summary += search_results[0]["error"] + "\n\n"
            else:
                summary = summarize_web_results(rephrased_query, search_results, conversation_manager)
                final_summary += summary + "\n\n"

        if final_summary:
            conversation_manager.add_interaction(original_query, final_summary)
            yield final_summary
        else:
            yield "Unable to generate a response. Please try a different query."
    else:
        # PDF-based logic
        try:
            embed = get_embeddings()
            if os.path.exists("faiss_database"):
                logging.info("Loading FAISS database...")
                database = FAISS.load_local("faiss_database", embed, allow_dangerous_deserialization=True)
                retriever = database.as_retriever(search_kwargs={"k": 20})
                
                all_relevant_docs = retriever.get_relevant_documents(message)
                relevant_docs = [
                    doc for doc in all_relevant_docs
                    if doc.metadata["source"] in selected_docs
                ]
                
                if not relevant_docs:
                    yield "No relevant information found in the selected documents."
                    return
    
                context_str = "\n".join([doc.page_content for doc in relevant_docs])
                logging.info(f"Context length: {len(context_str)}")
            else:
                yield "No documents available. Please upload PDF documents to answer questions."
                return
            
            # Check the model type
            if model.startswith("duckduckgo/"):
                # DuckDuckGo chat with context
                for partial_response in get_response_from_duckduckgo(message, model, context_str, num_calls, temperature):
                    yield partial_response
            elif model == "@cf/meta/llama-3.1-8b-instruct":
                # Cloudflare API
                for partial_response in get_response_from_cloudflare(
                    prompt="",
                    context=context_str,
                    query=message,
                    num_calls=num_calls,
                    temperature=temperature,
                    search_type="pdf"
                ):
                    yield partial_response
            else:
                # Hugging Face API
                for partial_response in get_response_from_pdf(
                    message, model, selected_docs, num_calls=num_calls, temperature=temperature
                ):
                    yield partial_response
        except Exception as e:
            logging.error(f"Error with {model}: {str(e)}")
            if "microsoft/Phi-3-mini-4k-instruct" in model:
                logging.info("Falling back to Mistral model due to Phi-3 error.")
                fallback_model = "mistralai/Mistral-7B-Instruct-v0.3"
                yield from respond(message, history, fallback_model, temperature, num_calls, selected_docs)
            else:
                yield f"An error occurred with {model}: {str(e)}. Try again or select a different model."

# --------------------------------------------------------------------------------
# Supporting Functions for PDF and Cloudflare
# --------------------------------------------------------------------------------

def get_response_from_cloudflare(prompt, context, query, num_calls=3, temperature=0.2, search_type="pdf"):
    """
    Utilize the Cloudflare Llama-based API to generate responses in streaming fashion.
    """
    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json"
    }
    model = "@cf/meta/llama-3.1-8b-instruct"

    if search_type == "pdf":
        instruction = f"""Using the following context from the PDF documents:
{context}
Write a detailed and complete response that answers the user question: '{query}'"""
    else:  # web search
        instruction = f"""Using the following context:
{context}
Write a detailed article that fulfills the user request: '{query}'
Then provide a list of sources used."""

    inputs = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": query}
    ]

    payload = {
        "messages": inputs,
        "stream": True,
        "temperature": temperature,
        "max_tokens": 32000
    }

    full_response = ""
    for i in range(num_calls):
        try:
            with requests.post(f"{API_BASE_URL}{model}", headers=headers, json=payload, stream=True) as response:
                if response.status_code == 200:
                    for line in response.iter_lines():
                        if line:
                            try:
                                json_response = json.loads(line.decode('utf-8').split('data: ')[1])
                                if 'response' in json_response:
                                    chunk = json_response['response']
                                    full_response += chunk
                                    yield full_response
                            except (json.JSONDecodeError, IndexError) as e:
                                logging.error(f"Error parsing streaming response: {str(e)}")
                                continue
                else:
                    logging.error(f"HTTP Error: {response.status_code}, response text: {response.text}")
                    yield f"HTTP error: {response.status_code}. Please try again later."
        except Exception as e:
            logging.error(f"Error in generating response from Cloudflare: {str(e)}")
            yield f"An error occurred: {str(e)}. Please try again later."

    if not full_response:
        yield "No response generated from Cloudflare."

def create_web_search_vectors(search_results):
    """
    Convert web search results into Document objects and build a FAISS index from them.
    """
    embed = get_embeddings()
    documents = []
    for result in search_results:
        if 'body' in result:
            content = f"{result['title']}\n{result['body']}\nSource: {result['href']}"
            documents.append(Document(page_content=content, metadata={"source": result['href']}))
    return FAISS.from_documents(documents, embed)

def get_response_from_pdf(query, model, selected_docs, num_calls=3, temperature=0.2):
    """
    Use a filtered FAISS index (based on selected_docs) to retrieve relevant PDF chunks,
    then generate responses from either Cloudflare or Hugging Face.
    """
    logging.info(f"get_response_from_pdf -> Query: {query}, Model: {model}, Docs: {selected_docs}")
    
    embed = get_embeddings()
    if not os.path.exists("faiss_database"):
        logging.warning("No FAISS database found.")
        yield "No documents available. Please upload PDF documents first."
        return

    database = FAISS.load_local("faiss_database", embed, allow_dangerous_deserialization=True)

    # Filter the docstore by user-selected documents
    filtered_docs = []
    for doc_id, doc in database.docstore._dict.items():
        if isinstance(doc, Document) and doc.metadata.get("source") in selected_docs:
            filtered_docs.append(doc)
    
    logging.info(f"Number of filtered documents: {len(filtered_docs)}")

    if not filtered_docs:
        yield "No relevant documents found among the selected sources."
        return

    # Build a new in-memory FAISS index with only the filtered docs
    filtered_db = FAISS.from_documents(filtered_docs, embed)
    retriever = filtered_db.as_retriever(search_kwargs={"k": 10})
    
    relevant_docs = retriever.get_relevant_documents(query)
    logging.info(f"Number of relevant documents: {len(relevant_docs)}")

    for doc in relevant_docs:
        logging.debug(f"Doc source: {doc.metadata['source']}, content preview: {doc.page_content[:100]}...")

    context_str = "\n".join([doc.page_content for doc in relevant_docs])
    logging.info(f"Total context length: {len(context_str)}")

    if model == "@cf/meta/llama-3.1-8b-instruct":
        logging.info("Using Cloudflare for PDF response.")
        for response in get_response_from_cloudflare(
            prompt="",
            context=context_str,
            query=query,
            num_calls=num_calls,
            temperature=temperature,
            search_type="pdf"
        ):
            yield response
    else:
        logging.info("Using Hugging Face for PDF response.")
        prompt = f"""Using the following context from the PDF documents:
{context_str}
Write a detailed and complete response that answers the user query: '{query}'"""

        client = InferenceClient(model, token=HUGGINGFACE_TOKEN)
        response_accumulator = ""

        for i in range(num_calls):
            logging.info(f"Hugging Face call {i+1} of {num_calls}")
            try:
                for message in client.chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=20000,
                    temperature=temperature,
                    stream=True
                ):
                    if (message.choices and 
                        message.choices[0].delta and 
                        message.choices[0].delta.content):
                        chunk = message.choices[0].delta.content
                        response_accumulator += chunk
                        yield response_accumulator
            except Exception as e:
                logging.error(f"Error from HF client: {str(e)}")
                yield f"Error occurred with Hugging Face inference: {str(e)}"

        logging.info("Completed Hugging Face PDF response.")

# --------------------------------------------------------------------------------
# Voting & Gradio Interface
# --------------------------------------------------------------------------------

def vote(data: gr.LikeData):
    """
    Handle upvote/downvote from Gradio.
    """
    action = "upvoted" if data.liked else "downvoted"
    logging.info(f"You {action} this response: {data.value}")

css = """
/* Fine-tune chatbox size */
.chatbot-container {
    height: 600px !important;
    width: 100% !important;
}
.chatbot-container > div {
    height: 100%;
    width: 100%;
}
"""

def initial_conversation():
    """
    Provide an initial system message or greeting in the chatbot.
    """
    return [
        (
            None,
            "Welcome! I'm your AI assistant for web search and PDF analysis.\n\n"
            "1. Toggle Web Search vs. PDF Search from Additional Inputs.\n"
            "2. Use web search to find new information.\n"
            "3. Upload PDFs, then ask me about their contents.\n"
            "4. For any issues or feedback, contact @desai.shreyas94.\n\n"
            "Let's get started! Upload some PDFs or ask a question."
        )
    ]

def refresh_documents():
    """
    Refresh the global 'uploaded_documents' list from local JSON and
    update the Gradio checkbox group.
    """
    global uploaded_documents
    uploaded_documents = load_documents_from_json()
    return display_documents()

# Initialize the global uploaded_documents from JSON on load
uploaded_documents = load_documents_from_json()

document_selector = gr.CheckboxGroup(label="Select documents to query")
use_web_search = gr.Checkbox(label="Use Web Search", value=False)
custom_placeholder = "Ask a question (toggle Web Search / PDF Chat in Additional Inputs)"

# --------------------------------------------------------------------------------
# Gradio Application
# --------------------------------------------------------------------------------

demo = gr.ChatInterface(
    fn=respond,
    additional_inputs_accordion=gr.Accordion(label="⚙️ Parameters", open=True, render=False),
    additional_inputs=[
        gr.Dropdown(choices=MODELS, label="Select Model", value=MODELS[3]),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.2, step=0.1, label="Temperature"),
        gr.Slider(minimum=1, maximum=5, value=1, step=1, label="Number of API Calls"),
        gr.Checkbox(label="Use Web Search", value=True),
        gr.CheckboxGroup(label="Select documents to query")
    ],
    title="AI-powered PDF Chat and Web Search Assistant",
    description="Chat with your PDFs or use web search to answer questions.",
    theme=gr.themes.Soft(
        primary_hue="orange",
        secondary_hue="amber",
        neutral_hue="gray",
        font=[gr.themes.GoogleFont("Exo"), "ui-sans-serif", "system-ui", "sans-serif"]
    ).set(
        body_background_fill_dark="#0c0505",
        block_background_fill_dark="#0c0505",
        block_border_width="1px",
        block_title_background_fill_dark="#1b0f0f",
        input_background_fill_dark="#140b0b",
        button_secondary_background_fill_dark="#140b0b",
        border_color_accent_dark="#1b0f0f",
        border_color_primary_dark="#1b0f0f",
        background_fill_secondary_dark="#0c0505",
        color_accent_soft_dark="transparent",
        code_background_fill_dark="#140b0b"
    ),
    css=css,
    examples=[
        ["Tell me about the contents of the uploaded PDFs."],
        ["What are the main topics discussed in the documents?"],
        ["Can you summarize the key points from the PDFs?"],
        ["What's the latest news about artificial intelligence?"]
    ],
    cache_examples=False,
    analytics_enabled=False,
    textbox=gr.Textbox(
        placeholder=custom_placeholder,
        container=False,
        scale=7
    ),
    chatbot=gr.Chatbot(
        show_copy_button=True,
        likeable=True,
        layout="bubble",
        height=400,
        value=initial_conversation()
    )
)

with demo:
    gr.Markdown("## Upload and Manage PDF Documents")
    with gr.Row():
        file_input = gr.Files(label="Upload your PDF documents", file_types=[".pdf"])
        parser_dropdown = gr.Dropdown(
            choices=["pypdf", "llamaparse"], label="Select PDF Parser", value="llamaparse"
        )
        update_button = gr.Button("Upload Document")
        refresh_button = gr.Button("Refresh Document List")
    
    update_output = gr.Textbox(label="Update Status")
    delete_button = gr.Button("Delete Selected Documents")
    
    # Upload / Update
    update_button.click(
        update_vectors,
        inputs=[file_input, parser_dropdown],
        outputs=[update_output, demo.additional_inputs[-1]]
    )
    # Refresh
    refresh_button.click(
        refresh_documents,
        inputs=[],
        outputs=[demo.additional_inputs[-1]]
    )
    # Delete
    delete_button.click(
        delete_documents,
        inputs=[demo.additional_inputs[-1]],
        outputs=[update_output, demo.additional_inputs[-1]]
    )

    gr.Markdown(
        """
        ## How to Use
        1. Upload PDFs, select the parser, and click "Upload Document" to update the vector store.
        2. Select the documents you want to query from the checkboxes.
        3. Ask questions in the chat interface.
        4. Toggle "Use Web Search" for external queries or keep it off for PDF-based responses.
        5. Adjust Temperature & API Calls for different response generation behaviors.
        6. Feel free to try the examples or ask your own questions!
        """
    )

if __name__ == "__main__":
    demo.launch(share=True)
