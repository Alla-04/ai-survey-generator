# --- 1. IMPORTS ---

import streamlit as st                                      # Streamlit for UI
import re                                                   # Regex for detecting commands like "more" or "about"
import requests                                             # Requests for calling Astra REST API
from langchain_openai import OpenAIEmbeddings, ChatOpenAI   # LangChain for LLM + embeddings + prompts
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import CharacterTextSplitter
from pypdf import PdfReader                                 # PyPDF to read the dataset file


# --- 2. SECRETS & INITIALIZATION ---

ASTRA_DB_APPLICATION_TOKEN = st.secrets["ASTRA_DB_APPLICATION_TOKEN"]   # Used to authenticate requests to AstraDB
ASTRA_DB_ENDPOINT = st.secrets["ASTRA_DB_ENDPOINT"]                     # Used as the REST API URL for connecting to AstraDB
ASTRA_DB_KEYSPACE = st.secrets["ASTRA_DB_KEYSPACE"]                     # Used to specify which keyspace to store/search data in
ASTRA_COLLECTION = st.secrets["ASTRA_COLLECTION"]                       # Used to define which table holds the embeddings
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]                           # Used to access OpenAI for embeddings and LLM generation

# LLM and embedding model
llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o-mini")
embedding = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model="text-embedding-3-small")


# --- 3. ASTRA REST API HELPERS ---

# Inserts a text chunk + embedding into Astra
def astra_insert_chunk(idx, text):
    # URL of the Astra REST API endpoint
    url = f"{ASTRA_DB_ENDPOINT}/api/json/v1/{ASTRA_DB_KEYSPACE}/{ASTRA_COLLECTION}"

    # Headers contain authentication + content type
    headers = {
        "x-cassandra-token": ASTRA_DB_APPLICATION_TOKEN,
        "Content-Type": "application/json"
    }

    # Payload includes: unique ID, chunk text, and its embedding vector
    payload = {
        "documentId": f"chunk_{idx}",
        "document": {
            "text": text,
            "embedding": embedding.embed_query(text)
        }
    }

    # Send POST request to Astra (store the document)
    requests.post(url, headers=headers, json=payload)


# Searches for similar embeddings (vector search)
def astra_query(top_k, query_text):
    # URL for vector search
    url = f"{ASTRA_DB_ENDPOINT}/api/json/v1/{ASTRA_DB_KEYSPACE}/{ASTRA_COLLECTION}/vector-search"

    headers = {
        "x-cassandra-token": ASTRA_DB_APPLICATION_TOKEN,
        "Content-Type": "application/json"
    }

    # Generate embedding for the query text
    payload = {
        "vector": embedding.embed_query(query_text),
        "topK": top_k
    }

    # Make the request and parse JSON response
    res = requests.post(url, headers=headers, json=payload).json()

    # Handle case where no documents exist yet
    if "documents" not in res:
        return ""

    # Return only the text fields of the retrieved chunks
    return "\n".join([doc["data"]["text"] for doc in res["documents"]])


# --- 4. PDF INGESTION ---

@st.cache_resource
def load_data():
    # Try loading the PDF. If missing then stop app.
    try:
        pdfreader = PdfReader("AI Survey Generator.pdf")
    except FileNotFoundError:
        st.error("ERROR: PDF not found. Upload it to your Streamlit app directory.")
        return None

    # Read every page in the PDF and combine all text into one big string
    raw_text = ""
    for page in pdfreader.pages:
        content = page.extract_text()
        if content:
            raw_text += content

    # Create a splitter to break the PDF into smaller chunks for better retrieval
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=800,
        chunk_overlap=200,
    )
    # Split the raw PDF text into multiple manageable pieces
    chunks = text_splitter.split_text(raw_text)

    # Insert stored chunks into Astra
    for i, t in enumerate(chunks[:50]):
        astra_insert_chunk(i, t)

    return "OK"

# Run PDF ingestion
if load_data() is None:
    st.stop()


# --- 5. RAG PROMPT ---

# This system instruction tells the LLM how it should behave.
# It forces the model to:
# - Follow the user’s requested format (MCQ, yes/no, etc.)
# - Use retrieved context only as inspiration
# - Output ONLY the list of questions (no explanations)
system_instruction = (
    "You are an expert survey question generator. Your primary goal is to fulfill the user's request exactly. "
    "1. Format: You MUST follow all specified numbers and types (MCQ, yes/no, rating scale, open-ended, Likert, Etc). "
    "2. Context: Use the retrieved context for inspiration only. Do not treat it as a restriction. "
    "3. Output: Return ONLY the generated list of questions. Do NOT include any introductory or explanatory text."
)

# Prompt template that combines:
# - The system instruction (role & rules)
# - The retrieved context from Astra
# - The specific user request (instruction)
prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_instruction),
    ("human", "Context:\n{context}\n\nUser Request: {instruction}")
])


# --- 6. UI + SMART COMMAND HANDLING ---

# Keywords that suggest the user wants a continuation based on the last topic
CONTINUATION_KEYWORDS = ["more", "additional", "another", "next", "same"]

# Memory storage for last topic
if "LAST_TOPIC" not in st.session_state:
    st.session_state.LAST_TOPIC = None

st.title("📝 Survey Question Generator")
st.write("Generate smart, high-quality survey questions instantly using AI. Enter any topic or format request (MCQ, yes/no, rating, open-ended, Likert, Etc) and the system will create tailored questions for you.")
user_input = st.text_input("Enter your request:")

if st.button("Generate"):

    # Basic safety check
    if not user_input:
        st.warning("Please enter your topic first.")
        st.stop()

    normalized_input = user_input.lower()   # Lowercased version for easier pattern checks
    topic_for_retrieval = None              # Which topic to search for in AstraDB
    llm_instruction = None                  # What instruction we send to the LLM

    # --- Rule 1: Detect CONTINUATION commands like “2 more”, “more questions” ---
    # Detect if the input contains a continuation keyword AND does NOT specify a new topic
    is_continuation_command = (
        any(kw in normalized_input for kw in CONTINUATION_KEYWORDS)
        and not re.search(r"about\s+[^,]+", normalized_input)
    )

    # If we already have a topic and user says a continuation keyword
    if st.session_state.LAST_TOPIC and is_continuation_command:

        # Continue with the previous topic 
        topic_for_retrieval = st.session_state.LAST_TOPIC

        # Simple continuation keywords should be 3 words or less (eg "more" or "give me more") and should not include a number
        simple_continuation = (
            any(kw in normalized_input for kw in CONTINUATION_KEYWORDS)
            and len(normalized_input.split()) <= 3
            and not re.search(r"\d", normalized_input)
        )

        # If the user types a simple continuation keyword then generate 5 new questions automatically
        if simple_continuation:
            llm_instruction = (
                f"Generate 5 NEW and DIFFERENT survey questions on the topic '{topic_for_retrieval}'. "
                "Ensure they are not duplicates of the previously generated set."
            )
        else:
            # If the user typed something more specific like "give me 3 more mcq"
            # We pass their exact request to the LLM and ensure questions are new
            llm_instruction = (
                f"Apply the following request to '{topic_for_retrieval}': {user_input}. "
                "Generate only NEW and DIFFERENT questions."
                "Ensure they are not duplicates of the previously generated set."
            )

    # --- Rule 2: User provides a NEW topic ---
    else:
        # If the user asks for “more” before any topic was ever given, show an error
        if not st.session_state.LAST_TOPIC and is_continuation_command:
            st.error("Please specify a topic first before asking for more questions.")
            st.stop()

        # Try to extract a topic after "about"
        match = re.search(r"about\s+([^,]+)", normalized_input)

        if match:
            # Topic detected explicitly
            st.session_state.LAST_TOPIC = match.group(1).strip()
        else:
            # Assume whole input is the topic
            st.session_state.LAST_TOPIC = user_input.strip()

        topic_for_retrieval = st.session_state.LAST_TOPIC
        llm_instruction = user_input

        # If topic is too short (example: "sales") then generate a safe and structured default: 8 general survey questions
        if len(user_input.split()) <= 2:
            llm_instruction = (
                f"Generate exactly 8 survey questions about {user_input}. "
                "Do not include any intro text."
            )
    
    # RAG query Astra for context
    context = astra_query(3, topic_for_retrieval)

    # Show spinner while generating
    with st.spinner("Generating your survey questions..."):
        response = llm.invoke(
            prompt_template.format(
                context=context,
                instruction=llm_instruction
            )
        )
        result = response.content

    st.subheader("Generated Questions")
    st.write(result)
