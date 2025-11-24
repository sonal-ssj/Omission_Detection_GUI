
# 🏥 MedOmitDetect: Medical AI Omission Detection System

End-to-end system for evaluating medical AI responses using Modal-hosted LLMs and OpenAI-powered omission detection with an interactive Gradio interface.

## 🎯 Overview

Deploy multiple medical language models (Llama3-OpenBioLLM, BioMistral, Meditron) via Modal's serverless platform and analyze responses for clinical completeness using GPT-4o. Categorizes detected omissions by clinical harm level (Mild, Moderate, Severe, Life-threatening).


## ✅ Prerequisites

- Python 3.9+
- Conda environment manager
- [Modal account](https://modal.com) (free tier available)
- [HuggingFace account](https://huggingface.co) with access to gated models
- OpenAI API key



## 📦 Installation

**Clone and set up environments:**

```bash
# Clone repository
git clone git@github.com:sonal-ssj/MedOmitDetect-LLM_Omission_Detection_GUI.git
cd medical-omission-detection

# Modal environment (for LLM deployment)
conda env create -f environment_modal.yml

# Gradio environment (for GUI)
conda env create -f environment.yml
```


***

## ⚙️ Configuration

### 1. Modal Setup

```bash
# Install and authenticate
conda activate vllm-modal-dev
modal setup
```


### 2. HuggingFace Token

```bash
# Create Modal secret for gated models
modal secret create huggingface-secret HF_TOKEN=hf_your_token_here
```

Get your token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and accept license agreements for Llama models.

### 3. OpenAI API Key

```bash
# Create .env file
cp example.env .env
# Add: OPENAI_API_KEY=sk-your_key_here
```


***

## 🚀 Deployment

**Deploy Modal LLM service:**

```bash
conda activate vllm-modal-dev
modal deploy modal_models.py
```

**Verify deployment:**

```bash
modal app list
# Look for: medical-llm-inference
```


***

## 💻 Usage

**Launch Gradio interface:**

```bash
conda activate medexpert_detector_gui
python omission_gui.py
# Access at: http://localhost:8181
```


**Workflow:**
1. Select medical question (predefined or custom)
2. Choose response model (8B for speed, 70B for accuracy)
3. Choose detection model (gpt-4o recommended)
4. Enter OpenAI API key if not in `.env`
5. Click "🔍 Analyze Response for Omissions"
6. Review results with severity classifications

***

## 🐛 Troubleshooting

**Modal connection fails:** Run `modal setup` and `modal deploy modal_models.py`

**HuggingFace access denied:** Accept model licenses and verify token permissions

**OpenAI API errors:** Check `.env` file and account billing status


***

## 📁 Project Structure

```
medical-omission-detection/
├── modal_models.py          # Modal LLM deployment
├── omission_gui.py           # Gradio interface
├── omission_grader.py        # Detection logic
├── environment_modal.yml     # Modal environment
├── environment.yml           # Gradio environment
└── example.env               # Environment template
```


***

## 📊 Available Models

**Fast (8B/7B parameters):** Llama3-OpenBioLLM-8B, BioMistral-7B, Meditron-7B, Meta-Llama-3-8B-Instruct

**Accurate (70B parameters):** Llama3-OpenBioLLM-70B, Meditron-70B, Meta-Llama-3-70B-Instruct

**Other:** MedGemma-27B

***

## 💡 Best Practices

- Start with 8B models for testing
- Keep Modal app warm with regular requests

## How to cite
If you use the MedExpert dataset or reference the related paper in your research, please cite the following work:

```
@inproceedings{Yarmohammadi2025MedExpert,
  title = {MedExpert: An Expert-Annotated Dataset for Medical Chatbot Evaluation},
  author = {Mahsa Yarmohammadi and Alexandra DeLucia and Lillian C. Chen and Leslie Miller and Heyuan Huang and Sonal Joshi and Jonathan Lasko and Sarah Collica and Ryan Moore and Haoling Qiu and Peter P. Zandi and Damianos Karakos and Mark Dredze},
  booktitle = {Proceedings of Machine Learning for Health (ML4H)},
  year = {2025}
}
```
