import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os
import torch

# Improved Model Loading with Medical Focus
@st.cache_resource
def load_model():
    model_path = "./my_gpt2_model"
    os.makedirs(model_path, exist_ok=True)
    
    try:
        # Check for required model files
        required_files = ['config.json', 'pytorch_model.bin']  # Changed from training_args.bin
        model_exists = all(os.path.exists(os.path.join(model_path, f)) for f in required_files)
        
        if not model_exists:
            model = GPT2LMHeadModel.from_pretrained("gpt2")
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            
            # Critical medical configuration
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.add_special_tokens({
                'additional_special_tokens': [
                    '[MED]', '[SYMPTOM]', '[TREATMENT]', '[QUESTION]'
                ]
            })
            model.resize_token_embeddings(len(tokenizer))
            
            model.save_pretrained(model_path)
            tokenizer.save_pretrained(model_path)
            return model, tokenizer, True  
        else:
            model = GPT2LMHeadModel.from_pretrained(model_path)
            tokenizer = GPT2Tokenizer.from_pretrained(model_path)
            return model, tokenizer, False
        
    except Exception as e:
        st.error(f"Model loading error: {str(e)}")
        return None, None, False

# Initialize model
model, tokenizer, is_new_download = load_model()

# UI Setup (unchanged from your original)
st.set_page_config(
    page_title="CycleCare Assistant",
    page_icon="🌸",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS (unchanged from your original)
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
    .quick-question-btn {
        width: 100%;
        margin-bottom: 10px;
    }
    .assistant-msg {
        background-color: #FFF5F5;
        padding: 12px;
        border-radius: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar (unchanged from your original)
with st.sidebar:
    st.header("🌸 CycleCare Guide")
    st.markdown("""
    <div class="instruction-box">
    <h4>How to use:</h4>
    <ol>
        <li>Type your question in the chat box</li>
        <li>Press Enter or click Send</li>
        <li>Try sample questions below</li>
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
    Not medical advice. Consult a doctor for:
    - Severe pain
    - Unusual bleeding
    - Persistent symptoms
    """)

# Main Chat Interface (unchanged from your original)
st.title("🌸 CycleCare Assistant")
st.markdown("### Your personal menstrual health guide")

# Initialize chat history (unchanged from your original)
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi there! I'm here to help with your menstrual health questions. What would you like to know today?"}
    ]

# Display chat messages (unchanged from your original)
for msg in st.session_state.messages:
    avatar = "🌸" if msg["role"] == "assistant" else "👩"
    st.chat_message(msg["role"], avatar=avatar).write(msg["content"])

# Medical Response Generation Parameters
MEDICAL_GENERATION_CONFIG = {
    "max_length": 200,
    "do_sample": True,
    "temperature": 0.7,
    "top_k": 50,
    "top_p": 0.9,
    "repetition_penalty": 1.5,
    "num_beams": 3,
    "no_repeat_ngram_size": 3,
    "pad_token_id": tokenizer.eos_token_id if tokenizer else 50256
}

def generate_medical_response(prompt):
    try:
        # Format the prompt with medical context
        formatted_prompt = f"[QUESTION] {prompt} [MED]"
        
        inputs = tokenizer(
            formatted_prompt,
            return_tensors="pt",
            max_length=100,
            truncation=True,
            padding=True
        )
        
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            **MEDICAL_GENERATION_CONFIG
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean up response
        response = response.replace(formatted_prompt, "").strip()
        response = response.split('[MED]')[0].split('[SYMPTOM]')[0].strip()
        
        return response
        
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Chat input
if prompt := st.chat_input("Type your question here..."):
    if not model or not tokenizer:
        st.error("Model not loaded. Please refresh the page.")
        st.stop()
        
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user", avatar="👩").write(prompt)
    
    # Generate response
    with st.spinner("Generating response..."):
        response = generate_medical_response(prompt)
        
        # Safety enhancements
        medical_keywords = {
            "emergency": ["severe pain", "heavy bleeding", "fever", "can't keep down"],
            "warning": ["ibuprofen", "missed period", "blood clots", "irregular"]
        }
        
        for level, terms in medical_keywords.items():
            if any(term in prompt.lower() for term in terms):
                if level == "emergency":
                    response += "\n\n🚨 **Important**: Please consult a doctor immediately for these symptoms"
                else:
                    response += "\n\n⚠️ **Note**: Consider discussing this with your healthcare provider"
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.chat_message("assistant", avatar="🌸").write(response)

# Quick question buttons (unchanged from your original)
st.markdown("### Common Questions")
cols = st.columns(3)
questions = [
    "What are common PMS symptoms?",
    "How can I relieve menstrual cramps?",
    "What's a normal menstrual cycle length?"
]

for col, question in zip(cols, questions):
    with col:
        if st.button(question, key=question, help=f"Ask: {question}"):
            st.session_state.messages.append({"role": "user", "content": question})
            st.rerun()