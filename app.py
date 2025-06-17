
import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load model
@st.cache_resource
def load_model():
    model = GPT2LMHeadModel.from_pretrained("Summative-ML-Technique-I-Chatbot
/my_gpt2_model")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    return model, tokenizer

model, tokenizer = load_model()

# UI Setup
st.set_page_config(
    page_title="CycleCare Assistant",
    page_icon="🌸",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stChatInput input {
        background-color: #FFF5F5 !important;
    }
    .stChatMessage {
        padding: 12px 16px;
        border-radius: 15px;
    }
    [data-testid="stSidebar"] {
        background-color: #FFF0F5;
    }
    .instruction-box {
        padding: 15px;
        background-color: #FFF5F5;
        border-radius: 10px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar - Enhanced Instructions
with st.sidebar:
    st.header("🌸 CycleCare Guide")
    st.markdown("""
    <div class="instruction-box">
    <h4>How to use:</h4>
    <ol>
        <li>Type your question in the chat box below</li>
        <li>Press Enter or click Send</li>
        <li>For best results, try:</li>
        <ul>
            <li>"What are PMS symptoms?"</li>
            <li>"How to relieve cramps naturally?"</li>
            <li>"Signs of ovulation"</li>
        </ul>
    </ol>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    st.markdown("""
    **Available Features:**
    - Period tracking advice
    - Symptom explanations
    - Pain management tips
    - Cycle education
    """)
    
    st.divider()
    st.warning("""
    **Disclaimer**  
    This is an AI assistant, not a medical professional.  
    Always consult your doctor for:
    - Severe pain
    - Unusual bleeding
    - Persistent symptoms
    """)

# Main Chat Interface
st.title("🌸 CycleCare Assistant")
st.markdown("### Your personal menstrual health guide")

# Welcome message
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi there! I'm here to help with your menstrual health questions. What would you like to know today?"}
    ]

# Display chat history
for msg in st.session_state.messages:
    avatar = "🌸" if msg["role"] == "assistant" else "👩"
    st.chat_message(msg["role"], avatar=avatar).write(msg["content"])

# Input and generation
if prompt := st.chat_input("Type your question here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user", avatar="👩").write(prompt)
    
    # Generate response
    with st.spinner("Thinking..."):
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(
            **inputs,
            max_length=200,
            do_sample=True,
            temperature=0.7,
            top_k=50
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Safety filters
        safety_flags = {
            "emergency": ["severe pain", "heavy bleeding", "fever"],
            "warning": ["ibuprofen", "missed period", "blood clots"]
        }
        
        for level, terms in safety_flags.items():
            if any(term in prompt.lower() for term in terms):
                if level == "emergency":
                    response += "\n\n🚨 **Emergency Alert**: Please seek immediate medical attention for these symptoms"
                else:
                    response += "\n\n⚠️ **Medical Notice**: Consult your healthcare provider about this"

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant", avatar="🌸").write(response)

# Quick question buttons
st.markdown("### Common Questions")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("PMS Symptoms"):
        st.session_state.messages.append({"role": "user", "content": "What are common PMS symptoms?"})
with col2:
    if st.button("Cramp Relief"):
        st.session_state.messages.append({"role": "user", "content": "How can I relieve menstrual cramps?"})
with col3:
    if st.button("Cycle Length"):
        st.session_state.messages.append({"role": "user", "content": "What's a normal menstrual cycle length?"})
