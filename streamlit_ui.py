import streamlit as st
import requests

API_URL = "http://127.0.0.1:8002/execute" 

st.title("🩺 Doctor Appointment System")

# Initialize session state for conversation history
if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "user_id" not in st.session_state:
    st.session_state.user_id = ""

# User ID input (only show if not set)
if not st.session_state.user_id:
    user_id = st.text_input("Enter your ID number:", "")
    if user_id:
        st.session_state.user_id = user_id
        st.rerun()
else:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.info(f"User ID: {st.session_state.user_id}")
    with col2:
        if st.button("Change ID"):
            st.session_state.user_id = ""
            st.session_state.conversation = []
            st.rerun()

# Display conversation history
if st.session_state.conversation:
    for msg in st.session_state.conversation:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.write(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.write(msg["content"])

# Input for new message
if st.session_state.user_id:
    # Use chat input for better UX
    user_query = st.chat_input("Type your message here...")
    
    if user_query:
        # Add user message to conversation
        st.session_state.conversation.append({"role": "user", "content": user_query})
        
        # Prepare request with conversation history
        conversation_history = [
            {"role": msg["role"], "content": msg["content"]} 
            for msg in st.session_state.conversation[:-1]  # Exclude current message
        ]
        
        try:
            with st.spinner("Processing..."):
                response = requests.post(
                    API_URL, 
                    json={
                        'message': user_query,
                        'id_number': int(st.session_state.user_id),
                        'conversation_history': conversation_history
                    },
                    verify=False
                )
                
            if response.status_code == 200:
                result = response.json()
                
                # Add assistant response to conversation
                if "response" in result:
                    assistant_response = result["response"]
                    st.session_state.conversation.append({"role": "assistant", "content": assistant_response})
                    
                    # Update conversation history from API if provided
                    if "conversation_history" in result:
                        # Sync with API's version (in case it's different)
                        pass
                    
                    st.rerun()  # Refresh to show new message
                else:
                    st.error("No response received from API")
            else:
                st.error(f"Error {response.status_code}: Could not process the request.")
        except Exception as e:
            st.error(f"Exception occurred: {e}")
    
    # Clear conversation button
    if st.button("Clear Conversation"):
        st.session_state.conversation = []
        st.rerun()