import os
import streamlit as st
import toml
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer

# Streamlit page configuration - MUST be the first Streamlit command
st.set_page_config(
    page_title="DebateBot",
    layout="centered"
)

# Custom CSS for dark theme and neon blue effects
st.markdown("""
    <style>
        /* Background color for the entire app */
        .main {
            background-color: #000000;
            color: white;
        }

        /* Title styling with neon blue effect */
        h1, h2, h3, h4, h5, h6 {
            color: white;
            text-shadow: 0 0 10px #00FFFF, 0 0 20px #00FFFF, 0 0 30px #00FFFF, 0 0 40px #00FFFF, 0 0 50px #00FFFF;
        }

        /* Chat input styling */
        .stTextInput > div > div > input {
            background-color: #1a1a1a;
            color: white;
            border: 1px solid #00FFFF;
            border-radius: 10px;
        }

        /* Chat message background and text color */
        .stChatMessage {
            background-color: #1a1a1a;
            color: white;
            border: 1px solid #00FFFF;
            border-radius: 10px;
            padding: 10px;
            margin-bottom: 10px;
        }

        /* Customize the markdown text */
        .stMarkdown p {
            color: white;
        }

        /* Styling for the chat input container */
        .stChatInput > div {
            background-color: #1a1a1a;
            border: 1px solid #00FFFF;
            border-radius: 10px;
            padding: 10px;
        }

        /* Styling for the chat input text area */
        .stChatInput > div > div > textarea {
            background-color: #1a1a1a;
            color: white;
            border: none;
            outline: none;
            box-shadow: none;
        }

        /* Placeholder text color in the chat input */
        ::placeholder {
            color: #cccccc;
        }

        /* Scrollbar styling */
        ::-webkit-scrollbar {
            width: 10px;
        }

        ::-webkit-scrollbar-thumb {
            background-color: #00FFFF;
            border-radius: 5px;
        }

        ::-webkit-scrollbar-track {
            background-color: #1a1a1a;
        }
    </style>
""", unsafe_allow_html=True)

working_dir = os.path.dirname(os.path.abspath(__file__))
config_data = toml.load(f"{working_dir}/configuration.toml")

GROQ_API_KEY = config_data["GROQ_API_KEY"]

# Save the API key to the environment variable
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

client = Groq()

# Initialize chat history as streamlit session state if not present already
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Streamlit page title
st.title("Debate Bot")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Define the system prompt
systemPrompt = (
    "You are a Tournament of Champions level LD and Policy debater and have been coaching for the past "
    "5 years now. Your students have had enormous success on the debate circuit. Based on the "
    "context I provide, answer any questions as a personal assistant."
)

# Load the knowledge base into memory with utf-8 encoding
knowledge_base_path = os.path.join(working_dir, 'debate_data.txt')
with open(knowledge_base_path, 'r', encoding='utf-8') as f:
    knowledge_base = f.read()

# Function to extract important words from the user prompt
def extract_keywords(prompt, top_n=5):
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform([prompt])
    feature_names = vectorizer.get_feature_names_out()
    dense = vectors.todense().tolist()[0]
    keywords = sorted(zip(feature_names, dense), key=lambda x: x[1], reverse=True)[:top_n]
    return [kw[0] for kw in keywords]

# Function to perform retrieval-augmented generation (RAG)
def perform_rag(query, knowledge_base, additional_contexts=None):
    keywords = extract_keywords(query)

    # Search the knowledge base using the extracted keywords
    contexts = []
    for line in knowledge_base.splitlines():
        if any(keyword in line.lower() for keyword in keywords):
            contexts.append(line)
        if len(contexts) >= 5:  # Limit to top 5 matches
            break

    # Add additional contexts if provided
    if additional_contexts:
        contexts.extend(additional_contexts)

    # Summarize or truncate contexts if needed
    summarized_contexts = [ctx[:500] + '...' if len(ctx) > 500 else ctx for ctx in contexts]

    augmented_query = "<CONTEXT>\n" + "\n\n-------\n\n".join(
        summarized_contexts[:5]) + "\n-------\n</CONTEXT>\n\nMY QUESTION:\n" + query

    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": systemPrompt},
            {"role": "user", "content": augmented_query}
        ]
    )
    return res.choices[0].message.content

# Input field for user's message
user_prompt = st.chat_input("Ask Debate Bot")

if user_prompt:
    # Check for large inputs
    if len(user_prompt) > 1000:
        st.error("Your prompt is too long. Please shorten it.")
    else:
        st.chat_message("user").markdown(user_prompt)
        st.session_state.chat_history.append({"role": "user", "content": user_prompt})

        # Perform RAG using the user prompt and knowledge base
        assistant_response = perform_rag(user_prompt, knowledge_base)

        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

        # Display the LLM's response
        with st.chat_message("assistant"):
            st.markdown(assistant_response)
