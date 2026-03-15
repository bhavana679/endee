import streamlit as st
import requests
import json
import time

# Page Configuration
st.set_page_config(
    page_title="AI Engineering Knowledge Engine",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Theme/Style
st.markdown("""
    <style>
    .main {
        background-color: #0f1117;
        color: #e0e0e0;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #4f46e5;
        color: white;
    }
    .stTextInput>div>div>input {
        background-color: #1e293b;
        color: white;
    }
    .source-tag {
        background-color: #334155;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8em;
        margin-right: 5px;
    }
    .chat-bubble {
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .user-bubble {
        background-color: #1e293b;
        border-left: 5px solid #4f46e5;
    }
    .ai-bubble {
        background-color: #0f172a;
        border-left: 5px solid #10b981;
    }
    </style>
""", unsafe_allow_html=True)

# Constants
API_BASE_URL = "http://localhost:8000"

# Sidebar: Management & Settings
with st.sidebar:
    st.title("Management")
    st.markdown("---")
    
    st.subheader("Database Actions")
    if st.button("Trigger Data Ingestion"):
        with st.spinner("Processing documents and generating embeddings..."):
            try:
                response = requests.post(f"{API_BASE_URL}/ingest")
                if response.status_code == 200:
                    st.success("Ingestion pipeline completed successfully!")
                else:
                    st.error(f"Failed to trigger ingestion: {response.text}")
            except Exception as e:
                st.error(f"Connection error: {str(e)}")
    
    st.markdown("---")
    st.subheader("System Status")
    try:
        health_resp = requests.get(f"{API_BASE_URL}/health")
        if health_resp.status_code == 200:
            status = health_resp.json()
            if status.get("endee_connected"):
                st.success("Endee DB: Connected")
            else:
                st.warning("Endee DB: Disconnected")
            st.info(f"API: Online")
        else:
            st.error("Backend API: Offline")
    except:
        st.error("Backend API: Offline")

    st.markdown("---")
    st.caption("AI Engineering Knowledge Memory Engine v1.0")

# Main Layout
st.title("AI Engineering Knowledge Memory Engine")
st.markdown("Ask deep engineering questions based on your specialized documentation and architecture decisions.")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "response_time_seconds" in message:
            st.caption(f"Response time: {message['response_time_seconds']} seconds")
        if "sources" in message and message["sources"]:
            source_links = " ".join([f"<span class='source-tag'>[{s['id']}] {s['file']}</span>" for s in message["sources"]])
            st.markdown(f"**Sources referenced:** {source_links}", unsafe_allow_html=True)
            
        if "contexts" in message:
            with st.expander("View Retrieved Contexts"):
                for idx, ctx in enumerate(message["contexts"]):
                    st.markdown(f"**[Rank {ctx.get('rank', idx+1)}] Source:** `{ctx['source']}` | **Distance:** `{ctx['distance']:.4f}`")
                    st.info(ctx['text'])

# Chat Input
if prompt := st.chat_input("How does the modular architecture work?"):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call AI Backend
    with st.chat_message("assistant"):
        with st.spinner("Retrieving memory and generating answer..."):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/query",
                    json={"query": prompt, "top_k": 3}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "No answer generated.")
                    contexts = data.get("contexts", [])
                    sources = data.get("sources", [])
                    resp_time = data.get("response_time_seconds", 0.0)
                    
                    st.markdown(answer)
                    st.caption(f"Response time: {resp_time} seconds")
                    
                    if sources:
                        source_links = " ".join([f"<span class='source-tag'>[{s['id']}] {s['file']}</span>" for s in sources])
                        st.markdown(f"**Sources referenced:** {source_links}", unsafe_allow_html=True)
                    
                    if contexts:
                        with st.expander("View Retrieved Contexts"):
                            for idx, ctx in enumerate(contexts):
                                st.markdown(f"**[Rank {ctx.get('rank', idx+1)}] Source:** `{ctx['source']}` | **Distance:** `{ctx['distance']:.4f}`")
                                st.info(ctx['text'])
                    
                    # Store assistant message in history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer, 
                        "contexts": contexts,
                        "sources": sources,
                        "response_time_seconds": resp_time
                    })
                else:
                    st.error(f"Error from API: {response.text}")
                    
            except Exception as e:
                st.error(f"Failed to communicate with backend: {str(e)}")

# Footer
st.markdown("---")
st.caption("Library: Endee Vector Database, Sentence Transformers, and FastAPI.")
