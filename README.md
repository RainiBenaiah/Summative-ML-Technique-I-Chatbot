# Summative-ML-Technique-I-Chatbot

# CycleCare Assistant 

A period education chatbot designed to provide comprehensive, accessible, and stigma-free menstrual health education. Built using fine-tuned GPT-2 and deployed with Gradio for an intuitive user experience.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Performance Metrics](#performance-metrics)
- [Installation & Setup](#installation--setup)
- [Running the Chatbot](#running-the-chatbot)
- [Usage Examples](#usage-examples)
- [Model Architecture](#model-architecture)
- [Contributing](#contributing)
- [Acknowledgments](#acknowledgments)
- [Support](#support)
- [Disclaimer](#disclaimer)

---

## Overview

CycleCare Assistant addresses the critical gap in accessible period education by providing:

- Period Tracking Advice: Guidance on monitoring menstrual cycles and recognizing patterns  
- Symptom Explanations: Detailed explanations of common menstrual symptoms and their causes  
- Pain Management Tips: Evidence-based strategies for managing menstrual discomfort  
- Cycle Education: Comprehensive information about menstrual health and reproductive wellness  

### Target Audience

- Adolescents and young adults seeking period education  
- Individuals looking for reliable menstrual health information  
- Anyone wanting to understand menstrual cycles better  
- Partners and family members seeking to provide support  

---

## Dataset

### Dataset Overview

The model was trained on the Menstrual Health Awareness Dataset from Hugging Face: [`gjyotk/Menstrual-Health-Awareness-Dataset`](https://huggingface.co/datasets/gjyotk/Menstrual-Health-Awareness-Dataset)

**Statistics**:
- Training Samples: 530
- Test Samples: 45
- Total: 575 Q&A pairs
- Format: Conversational, age-appropriate, medically verified

### Data Characteristics

- Instruction Types: Periods, symptoms, pain management, cycle health  
- Coverage: PMS, cramps, ovulation, health tracking, education  
- Language: Plain, accessible English for diverse age groups  

### Preprocessing Steps

- Text normalization, contraction expansion  
- Medical term corrections  
- Unicode, whitespace, and punctuation cleanup  

---

## Performance Metrics

### Training Metrics

| Metric              | Value          |
|---------------------|----------------|
| Global Steps        | 1,700          |
| Training Duration   | 13h 25m        |
| Final Training Loss | 0.7047         |
| Throughput          | 65.666 samples/sec |
| Processing Speed    | 2.106 steps/sec   |

### Loss Progression

| Step  | Training Loss | Validation Loss |
|-------|---------------|-----------------|
| 100   | 1.4375        | 1.3033          |
| 500   | 0.4744        | 0.5721          |
| 1000  | 0.3269        | 0.5266          |
| 1500  | 0.2696        | 0.5286          |
| 1700  | 0.2641        | 0.5280          |

### Evaluation Metrics

- BLEU Score: 0.00 – 0.03 [(Hugging Face, n.d.)](https://huggingface.co/spaces/evaluate-metric/bleu)  
- F1 Score: 0.00 – 0.15 [(GeeksforGeeks, 2023)](https://www.geeksforgeeks.org/f1-score-in-machine-learning/)  
- Perplexity Range: 7.17 – 4093.52  

---
## Usage Examples

* image .*

### Conversation 1: Basic Period Information  
**User**: What is PMS?  
**Generated**: PMS is a chronic disorder characterized by excessive changes in menstrual hormone levels and changes in blood pressure and body temperature fluctuations due to hormonal fluctuations. Symptoms may include irregular periods, heavy menstrual bleeding, painful periods, and changes in libido or self-consciousness.

---

### Conversation 2: Pain Management  
**User**: How can I manage menstrual cramps?  
**Generated**: · Use medication to manage menstrual cramps, such as ibuprofen or acetaminophen, as needed to manage symptoms such as constipation and bloating.

---

### Input Tips

- Use natural language  
- Provide specific questions  
- Add context for better responses  
- Ask follow-up questions for more clarity if needed

## Installation & Setup

### Prerequisites

```bash
Python 3.8+
PyTorch
Transformers
Gradio
Pandas
NumPy

## Installation & Setup

To set up the CycleCare Assistant locally:

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/cyclecare-assistant.git
cd cyclecare-assistant

# 2. Create and activate a virtual environment
python -m venv chatbot
source chatbot/bin/activate  # On Windows: chatbot\Scripts\activate

# 3. Install required dependencies
pip install torch transformers gradio pandas numpy
pip install datasets tokenizers
pip install scikit-learn matplotlib seaborn

# 4. Download the fine-tuned model 
wget https://your-model-hosting-url/cyclecare-model.tar.gz
tar -xzf cyclecare-model.tar.gz

# 5. Run the chatbot
python app.py
---
## Contributing

We welcome community contributions!
---
### Steps

1. Fork the repository  
2. Create a branch: `git checkout -b feature/your-feature`  
3. Make changes and test  
4. Commit: `git commit -m "Your changes"`  
5. Push and open a PR  
---
### Contribution Areas

- More dataset samples  
- UI/UX improvements  
- Language translations  
- Mobile optimization  
---
---
---
---
## Acknowledgments

- Dataset: [`gjyotk/Menstrual-Health-Awareness-Dataset`](https://huggingface.co/datasets/gjyotk/Menstrual-Health-Awareness-Dataset)  
- Transformers: Hugging Face  
- Interface: Gradio  
- Educational inspiration: [WaterAid UK on Menstrual Hygiene](https://www.wateraid.org/uk/get-involved/teaching-resources/menstrual-hygiene/why-talk-menstrual-hygiene-in-the-classroom)

---

## Disclaimer

CycleCare Assistant is for educational use only. It should not replace professional medical consultation. Always consult a healthcare provider for medical concerns.

---

Made with care to empower period education and break the stigma.
----
## Support

- Email: raijustanobody@cyclecare.com  
- GitHub Issues: [Open an issue](https://github.com/yourusername/cyclecare-assistant/issues)  
- Discussions: [Join the conversation](https://github.com/yourusername/cyclecare-assistant/discussions)
----
