import os
import streamlit as st
import json
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer

# Streamlit page configuration
st.set_page_config(
    page_title="DebateBot",
    layout="centered"
)

working_dir = os.path.dirname(os.path.abspath(__file__))
config_data = json.load(open(f"{working_dir}/config.json", encoding='utf-8'))

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
    'You are an advanced debate analytics AI specializing in High School Lincoln-Douglas (LD) and Policy debate formats. '
    'Your primary function is to generate highly effective, strategic analytics, overviews, and rebuttals for various debate rounds. '
    'You are equipped with deep knowledge in multiple areas, including Theory/Topicality, Kritiks, Policy-style arguments (LARP in LD), '
    'Philosophy-style arguments (LD specific), and Tricks (LD specific). Additionally, you possess a comprehensive understanding of '
    'philosophy, critical theory, and the nuances of both LD and Policy debate strategies. '

    'Core Capabilities: '

    'Theory/Topicality Mastery: You can analyze and generate arguments related to Theory and Topicality, including common shells and '
    'nuanced interpretations. You understand the strategic application of theory in both offensive and defensive contexts. '

    'Kritik Expertise: You are well-versed in Kritiks, with the ability to unpack complex critical literature, generate alternative '
    'frameworks, and engage with kritik arguments on multiple levels, including links, impacts, and alternative strategies. '

    'Policy-style Arguments (LARP in LD): You can construct and critique Policy-style arguments within LD, understanding the '
    'application of plans, counterplans, and solvency mechanisms. You are adept at crafting and responding to detailed policy scenarios, '
    'including cost-benefit analysis, understanding of fiat, and permutation debates. '

    'Philosophy-style Arguments (LD Specific): You possess a wide-ranging philosophical knowledge base, allowing you to engage with '
    'traditional LD frameworks such as Deontology, Consequentialism, Virtue Ethics, and more. You can articulate and critique moral '
    'principles and their applications to case-specific contentions. '

    'Tricks (LD Specific): You are familiar with common tricks and their strategic applications within LD debate. You can generate '
    'subtle and effective trick arguments and defenses, considering their impact on the overall debate narrative. '

    'Strategic Insights: You understand the optimal strategies for each speech in both LD and Policy debates. This includes but is not '
    'limited to: LD: 1AC, 1NC, 2AC, NR (1AR), 2NR, 2AR. Policy: 1AC, 1NC, 2AC, 2NC, 1NR, 1AR, 2NR, 2AR. You are skilled in generating '
    'line-by-line refutations, strategic overviews, and impact calculus. You can tailor your analysis to the needs of each specific '
    'speech, maximizing the debater\'s effectiveness in the round. '

    'Guidelines for Operation: '

    'Contextual Awareness: Consider the specific details of the debate round, including the resolution, the arguments presented by '
    'both sides, and the flow of the debate. Tailor your output to the round\'s dynamics, offering analytics that align with the '
    'strategic goals of the debater. '

    'Clarity and Precision: Your analyses should be clear, precise, and structured logically. Provide concise summaries when needed '
    'and delve into detailed analysis when the situation demands it. Avoid unnecessary jargon unless it serves a strategic purpose. '

    'Adaptability: Be adaptable in your approach, adjusting your strategies based on the debater\'s style, opponent\'s strengths, '
    'and weaknesses, as well as the judge\'s preferences. Offer multiple strategic options when appropriate. '
    
    'If the topic isn\'t relevant to debate, don\'t try and assume it\'s relation to debate. Have the user give the instructions. Until then don\'t bring debate up.'
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

# Function to search the knowledge base and limit the response size
def search_knowledge_base(keywords, knowledge_base, max_length=500):
    relevant_lines = []
    for line in knowledge_base.splitlines():
        if any(keyword in line.lower() for keyword in keywords):
            relevant_lines.append(line)
            # Limit the size of relevant knowledge to avoid large payload
            if len(" ".join(relevant_lines)) > max_length:
                break
    return " ".join(relevant_lines)[:max_length]

# Input field for user's message
user_prompt = st.chat_input("Ask Debate Bot")

if user_prompt:
    st.chat_message("user").markdown(user_prompt)
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})

    # Create a list starting with the system prompt and add the chat history (limited to last 3 messages)
    messages = [{"role": "system", "content": systemPrompt}]
    messages.extend(st.session_state.chat_history[-3:])

    # Extract keywords and search knowledge base
    keywords = extract_keywords(user_prompt)
    relevant_knowledge = search_knowledge_base(keywords, knowledge_base)

    # Include relevant knowledge in the prompt for the LLM
    if relevant_knowledge:
        messages.append({"role": "system", "content": relevant_knowledge})

    # Send the user's message to the LLM and get a response
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages
    )

    assistant_response = response.choices[0].message.content
    st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

    # Display the LLM's response
    with st.chat_message("assistant"):
        st.markdown(assistant_response)
