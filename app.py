# --- 1. IMPORTS ---

import streamlit as st                                      # Streamlit for UI
import re                                                   # Regex for detecting commands like "more" or "about"
import requests                                             # Requests for calling Astra REST API
from langchain_openai import OpenAIEmbeddings, ChatOpenAI   # LangChain for LLM + embeddings + prompts
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import CharacterTextSplitter
from pypdf import PdfReader                                 # PyPDF to read the dataset file
import os                                                   # Environment variables for LangSmith setup
from langsmith import Client                                # Client for LLM tracing and evaluation


# --- 2. SECRETS & INITIALIZATION ---

ASTRA_DB_APPLICATION_TOKEN = st.secrets["ASTRA_DB_APPLICATION_TOKEN"]   # Used to authenticate requests to AstraDB
ASTRA_DB_ENDPOINT = st.secrets["ASTRA_DB_ENDPOINT"]                     # Used as the REST API URL for connecting to AstraDB
ASTRA_DB_KEYSPACE = st.secrets["ASTRA_DB_KEYSPACE"]                     # Used to specify which keyspace to store/search data in
ASTRA_COLLECTION = st.secrets["ASTRA_COLLECTION"]                       # Used to define which table holds the embeddings
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]                           # Used to access OpenAI for embeddings and LLM generation

# LangSmith configuration
os.environ["LANGCHAIN_TRACING_V2"] = "true"                        # Enables LangSmith tracing for all LangChain LLM calls
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGSMITH_API_KEY"]  # Authenticates the application with LangSmith
os.environ["LANGCHAIN_PROJECT"] = "AI-Survey-Generator"            # Groups all traces under a named LangSmith project

langsmith_client = Client()                                        # Initializes the LangSmith client for logging and evaluation


# LLM, embedding model, and LangSmith 
llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o-mini")
embedding = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model="text-embedding-3-small")
judge_llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o-mini")


# --- 3. ASTRA REST API HELPERS ---

# Inserts a text chunk + its embedding into AstraDB
def astra_insert_chunk(idx, text):
    # Build Astra endpoint URL (collection level)
    url = f"{ASTRA_DB_ENDPOINT}/api/json/v1/{ASTRA_DB_KEYSPACE}/{ASTRA_COLLECTION}"

    # Headers for authentication + JSON request
    headers = {
        "x-cassandra-token": ASTRA_DB_APPLICATION_TOKEN,
        "Content-Type": "application/json"
    }

    # Payload structure for inserting a document
    # Each chunk gets:
    # - unique ID
    # - raw text
    # - vector embedding (used later for similarity search)
    payload = {
        "insertOne": {
            "document": {
                "_id": f"chunk_{idx}",
                "text": text,
                "$vector": embedding.embed_query(text)
            }
        }
    }

    try:
        # Send insert request to Astra
        response = requests.post(url, headers=headers, json=payload, timeout=30)

        # If insert fails, show warning + response details
        if response.status_code not in [200, 201]:
            st.warning(f"Insert failed for chunk_{idx}: {response.status_code}")
            st.code(response.text)

    except requests.RequestException as e:
        # Handle network/API errors
        st.warning(f"Insert request failed for chunk_{idx}: {e}")


# Searches AstraDB for most similar chunks using vector search
def astra_query(top_k, query_text):
    # Same endpoint (query happens on collection)
    url = f"{ASTRA_DB_ENDPOINT}/api/json/v1/{ASTRA_DB_KEYSPACE}/{ASTRA_COLLECTION}"

    headers = {
        "x-cassandra-token": ASTRA_DB_APPLICATION_TOKEN,
        "Content-Type": "application/json"
    }

    # Convert user query into embedding vector
    query_vector = embedding.embed_query(query_text)

    # Payload:
    # - no filter (search everything)
    # - sort by similarity using 'vector' column
    # - limit results to top_k
    payload = {
        "find": {
            "filter": {},
            "sort": {
                "vector": query_vector
            },
            "options": {
                "limit": top_k
            }
        }
    }

    try:
        # Send search request to Astra
        response = requests.post(url, headers=headers, json=payload, timeout=30)

        # If API returns error (e.g., wrong schema / endpoint)
        if response.status_code != 200:
            st.error(f"Astra query failed: {response.status_code}")
            st.code(response.text)
            return ""

        # Handle empty response case
        if not response.text.strip():
            st.error("Astra returned an empty response.")
            return ""

        # Try parsing JSON safely
        try:
            res = response.json()
        except ValueError:
            st.error("Astra returned non-JSON content.")
            st.code(response.text)
            return ""

        # Extract documents list
        documents = res.get("data", {}).get("documents", [])

        # If nothing found, show debug info
        if not documents:
            st.warning("No documents found in Astra response.")
            st.json(res)
            return ""

        # Return only text content (used later in RAG prompt)
        return "\n".join(
            doc.get("text", "")
            for doc in documents
            if doc.get("text")
        )

    except requests.RequestException as e:
        # Handle connection errors
        st.error(f"Request to Astra failed: {e}")
        return ""

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
    "You are an expert survey question generator. Your primary goal is to fulfill the user's request exactly."
    "1. Format: You MUST follow all specified numbers and types (MCQ, yes/no, rating scale, open-ended, Likert, Etc)."
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


# --- 5.5 LLM-AS-JUDGE PROMPT ---

# Prompt used by the LLM-as-a-judge to evaluate relevance, instruction compliance, and cleanliness of the generated survey questions
judge_prompt = ChatPromptTemplate.from_template("""
You are an evaluation assistant.

User Request:
{user_input}

Generated Survey Questions:
{output}

Evaluate the output based on:
1. Topic relevance
2. Compliance with specific instructions (number, format, type)
3. Output cleanliness (no extra text, no duplicates)

Respond ONLY in this format:
Score: <0-100>
Reason: <very brief explanation>
""")


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
    context = astra_query(3, topic_for_retrieval) or ""

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


    # --- 7. LANGSMITH LLM-AS-JUDGE EVALUATION ---

    # Send the generated output to the judge LLM while showing a loading spinner
    with st.spinner("Evaluating AI output..."):
        judge_response = judge_llm.invoke(
            judge_prompt.format(
                user_input=user_input, # Original user request
                output=result          # AI-generated survey questions
            )
        )

    # Extract the text response from the judge LLM
    judge_output = judge_response.content

    # Extract the numeric score (0–100) from the judge response
    score_match = re.search(r"Score:\s*(\d+)", judge_output)
    judge_score = int(score_match.group(1)) if score_match else None

    # Display evaluation results in the UI
    st.subheader("Evaluation (LLM-as-a-Judge)")

    if judge_score is not None:
        st.metric("Judge Score", f"{judge_score}%")

        # Interpret the score using predefined thresholds
        if judge_score >= 85:
            st.success("Output strongly satisfies the user request.")
        elif judge_score >= 60:
            st.warning("Output partially satisfies the user request.")
        else:
            st.error("Output does not satisfy key requirements.")
    else:
        # Handle cases where the judge output format is invalid
        st.warning("Unable to extract judge score.")

    st.text_area("Judge Explanation", judge_output, height=150)
