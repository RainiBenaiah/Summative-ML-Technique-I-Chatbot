import gradio as gr
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os
import torch

# Improved Model Loading with Medical Focus
def load_model():
    model_path = "./my_gpt2_model"
    os.makedirs(model_path, exist_ok=True)
    
    try:
        # Check for required model files
        required_files = ['config.json', 'pytorch_model.bin']
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
        print(f"Model loading error: {str(e)}")
        return None, None, False

# Initialize model
model, tokenizer, is_new_download = load_model()

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

def generate_medical_response(prompt, history):
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
        
        return response
        
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Custom CSS for the app
css = """
.gradio-container {
    background-color: #FFF5F5 !important;
}
.chatbot {
    min-height: 500px;
}
.quick-questions {
    margin-top: 20px;
}
.quick-question-btn {
    width: 100%;
    margin-bottom: 10px;
}
.disclaimer {
    color: #ff4b4b;
    font-size: 0.9em;
    margin-top: 20px;
}
"""

# Quick question buttons
quick_questions = [
    "What are common PMS symptoms?",
    "How can I relieve menstrual cramps?",
    "What's a normal menstrual cycle length?"
]

# Create the chat interface
with gr.Blocks(css=css, title="🌸 CycleCare Assistant") as demo:
    gr.Markdown("## 🌸 CycleCare Assistant")
    gr.Markdown("### Your personal menstrual health guide")
    
    # Chatbot interface
    chatbot = gr.Chatbot(
        label="Chat History",
        avatar_images=("👩", "🌸"),
        bubble_full_width=False
    )
    
    # Initialize with welcome message
    def initialize_chat():
        return [[None, "Hi there! I'm here to help with your menstrual health questions. What would you like to know today?"]]
    
    demo.load(initialize_chat, None, chatbot)
    
    # Chat input
    msg = gr.Textbox(
        label="Type your question here...",
        placeholder="Ask about menstrual health...",
        container=False
    )
    
    # Clear button
    clear = gr.ClearButton([msg, chatbot])
    
    # Quick question buttons
    with gr.Row():
        for question in quick_questions:
            gr.Button(question).click(
                lambda q=question: q,
                outputs=msg
            )
    
    # Submit handler
    def respond(message, chat_history):
        if not model or not tokenizer:
            return "Model not loaded. Please refresh the page.", chat_history
            
        bot_message = generate_medical_response(message, chat_history)
        chat_history.append((message, bot_message))
        return "", chat_history
    
    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    
    # Sidebar content
    with gr.Accordion("🌸 CycleCare Guide", open=False):
        gr.Markdown("""
        **How to use:**
        1. Type your question in the chat box
        2. Press Enter or click Send
        3. Try sample questions below
        
        **Available Features:**
        - Period tracking advice
        - Symptom explanations
        - Pain management tips
        - Cycle education
        """)
        
        gr.Markdown("""
        <div class="disclaimer">
        **Disclaimer**  
        Not medical advice. Consult a doctor for:
        - Severe pain
        - Unusual bleeding
        - Persistent symptoms
        </div>
        """)

# Launch the app
if __name__ == "__main__":
    demo.launch()
