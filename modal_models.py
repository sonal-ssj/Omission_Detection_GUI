"""
VLLM Multi-Model Medical LLM Inference Server on Modal
"""

import modal
from typing import Dict, Optional, List

# Define Modal image
vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm==0.6.3.post1",
        "transformers>=4.45.0",
        "torch>=2.4.0",
        "accelerate>=0.34.0",
    )
)

app = modal.App("medical-llm-inference")

# Persistent cache volumes
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

# Model configurations
MODEL_CONFIGS = {
    "openbiollm_8b": {
        "model_id": "aaditya/Llama3-OpenBioLLM-8B",
        "gpu": "A10G",
        "tensor_parallel": 1,
        "max_model_len": 8192,
    },
    "llama3_8b": {
        "model_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "gpu": "A10G",
        "tensor_parallel": 1,
        "max_model_len": 8192,
    },
    "biomistral_7b": {
        "model_id": "BioMistral/BioMistral-7B",
        "gpu": "A10G",
        "tensor_parallel": 1,
        "max_model_len": 4096,
    },
    "meditron_7b": {
        "model_id": "epfl-llm/meditron-7b",
        "gpu": "A10G",
        "tensor_parallel": 1,
        "max_model_len": 4096,
    },
    "openbiollm_70b": {
        "model_id": "aaditya/Llama3-OpenBioLLM-70B",
        "gpu": "A100:2",
        "tensor_parallel": 2,
        "max_model_len": 8192,
    },
    "llama3_70b": {
        "model_id": "meta-llama/Meta-Llama-3-70B-Instruct",
        "gpu": "A100:2",
        "tensor_parallel": 2,
        "max_model_len": 8192,
    },
    "meditron_70b": {
        "model_id": "epfl-llm/meditron-70b",
        "gpu": "A100:2",
        "tensor_parallel": 2,
        "max_model_len": 4096,
    },
    "medgemma_27b": {
        "model_id": "google/medgemma-27b-it",
        "gpu": "A100",
        "tensor_parallel": 1,
        "max_model_len": 8192,
        "trust_remote_code": True,
    },
}

SYSTEM_PROMPT = """You are an AI-powered helpful health information assistant. Your goal is to provide health education. Explain medical concepts, conditions, treatments, and procedures in accurate, easy-to-understand language. Avoid overly technical jargon.

Scope of Your Function:
• Explain medical terminology and concepts.
• Summarize information about diagnosed conditions and standard treatment options.
• Provide overviews of medical procedures.
• Offer information on established public health guidelines."""


# Parametrized inference class for programmatic access
@app.cls(
    image=vllm_image,
    gpu="A10G",
    timeout=3600,
    scaledown_window=300,
    volumes={"/root/.cache/huggingface": hf_cache_vol},
)
@modal.concurrent(max_inputs=10)
class MedicalLLMInference:
    model_name: str = modal.parameter(default="openbiollm_8b")
    
    @modal.enter()
    def load_model(self):
        """Load VLLM model"""
        from vllm import LLM
        
        config = MODEL_CONFIGS[self.model_name]
        print(f"Loading model: {config['model_id']}")
        
        self.llm = LLM(
            model=config["model_id"],
            tensor_parallel_size=config.get("tensor_parallel", 1),
            max_model_len=config.get("max_model_len", 4096),
            trust_remote_code=config.get("trust_remote_code", False),
            dtype="bfloat16",
            gpu_memory_utilization=0.90,
            download_dir="/root/.cache/huggingface",
        )
        
        # Store tokenizer for chat template
        self.tokenizer = self.llm.get_tokenizer()
        
        print(f"Model {self.model_name} loaded successfully")
        
        # Check if chat template is available
        if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template:
            print(f"✓ Chat template available for {self.model_name}")
        else:
            print(f"⚠ No chat template found, using fallback formatting")
    
    @modal.method()
    def generate(
        self,
        user_prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        system_prompt: Optional[str] = None,
        debug: bool = False,
    ) -> Dict:
        """Generate response with proper chat template handling"""
        from vllm import SamplingParams
        
        sys_prompt = system_prompt or SYSTEM_PROMPT
        
        # Prepare messages for chat template
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Try to use chat template, fallback to manual formatting
        formatted_prompt = None
        template_used = "none"
        
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                template_used = "auto"
                if debug:
                    print(f"[DEBUG] Using automatic chat template")
            except Exception as e:
                if debug:
                    print(f"[DEBUG] Chat template error: {e}")
                formatted_prompt = None
        
        # Fallback: Manual Llama3 formatting for OpenBioLLM
        if formatted_prompt is None:
            if "llama3" in self.model_name.lower() or "openbiollm" in self.model_name.lower():
                formatted_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{sys_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
                template_used = "llama3_manual"
                if debug:
                    print(f"[DEBUG] Using manual Llama3 template")
            elif "mistral" in self.model_name.lower():
                formatted_prompt = f"<s>[INST] {sys_prompt}\n\n{user_prompt} [/INST]"
                template_used = "mistral_manual"
                if debug:
                    print(f"[DEBUG] Using manual Mistral template")
            else:
                # Generic fallback
                formatted_prompt = f"{sys_prompt}\n\nPatient Question: {user_prompt}\n\nAssistant:"
                template_used = "generic"
                if debug:
                    print(f"[DEBUG] Using generic template")
        
        if debug:
            print(f"[DEBUG] Template used: {template_used}")
            print(f"[DEBUG] Prompt preview (first 300 chars):\n{formatted_prompt[:300]}...")
            print(f"[DEBUG] Prompt length: {len(formatted_prompt)} characters")
        
        # Configure sampling parameters
        # More permissive stop tokens - let model decide when to stop
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=["<|eot_id|>", "<|end_of_text|>"],  # Only essential stop tokens for Llama3
        )
        
        # Generate
        outputs = self.llm.generate([formatted_prompt], sampling_params)
        
        # Extract response
        response_text = outputs[0].outputs[0].text.strip()
        finish_reason = outputs[0].outputs[0].finish_reason
        
        if debug:
            print(f"[DEBUG] Generated {len(outputs[0].outputs[0].token_ids)} tokens")
            print(f"[DEBUG] Finish reason: {finish_reason}")
            print(f"[DEBUG] Response preview: {response_text[:200]}...")
        
        return {
            "model": self.model_name,
            "text": response_text,
            "prompt_tokens": len(outputs[0].prompt_token_ids),
            "completion_tokens": len(outputs[0].outputs[0].token_ids),
            "finish_reason": finish_reason,
            "template_used": template_used,
        }
    
    @modal.method()
    def batch_generate(
        self,
        prompts: List[str],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        system_prompt: Optional[str] = None,
    ) -> List[Dict]:
        """Batch generate responses with proper chat template"""
        from vllm import SamplingParams
        
        sys_prompt = system_prompt or SYSTEM_PROMPT
        formatted_prompts = []
        
        # Format each prompt using chat template
        for prompt in prompts:
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt}
            ]
            
            try:
                formatted = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception:
                # Fallback for Llama3-based models
                formatted = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{sys_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
            formatted_prompts.append(formatted)
        
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=["<|eot_id|>", "<|end_of_text|>"],
        )
        
        outputs = self.llm.generate(formatted_prompts, sampling_params)
        
        return [
            {
                "model": self.model_name,
                "text": output.outputs[0].text.strip(),
                "prompt": prompts[i],
                "prompt_tokens": len(output.prompt_token_ids),
                "completion_tokens": len(output.outputs[0].token_ids),
                "finish_reason": output.outputs[0].finish_reason,
            }
            for i, output in enumerate(outputs)
        ]


# Parametrized OpenAI-compatible API server
@app.cls(
    image=vllm_image,
    volumes={"/root/.cache/huggingface": hf_cache_vol},
)
@modal.concurrent(max_inputs=100)
class VLLMServer:
    model: str = modal.parameter(default="openbiollm_8b")
    
    @modal.web_server(8000, startup_timeout=300)
    def serve(self):
        """Serve OpenAI-compatible API"""
        import subprocess
        import os
        
        config = MODEL_CONFIGS[self.model]
        os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
        
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", config["model_id"],
            "--tensor-parallel-size", str(config.get("tensor_parallel", 1)),
            "--max-model-len", str(config.get("max_model_len", 4096)),
            "--dtype", "bfloat16",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--gpu-memory-utilization", "0.90",
            "--served-model-name", self.model,  # Add model name
        ]
        
        if config.get("trust_remote_code", False):
            cmd.append("--trust-remote-code")
        
        print(f"Starting VLLM server for {self.model}: {config['model_id']}")
        subprocess.Popen(cmd)
    
    @modal.enter()
    def configure_gpu(self):
        """Log GPU configuration"""
        config = MODEL_CONFIGS[self.model]
        print(f"Using GPU: {config.get('gpu', 'A10G')}")


# Local testing entrypoint
@app.local_entrypoint()
def main(
    model: str = "openbiollm_8b",
    prompt: str = "What are the symptoms of Type 2 diabetes?",
    debug: bool = True,  # Enable debug by default for testing
):
    """Test the inference"""
    print(f"Testing model: {model}")
    print(f"Prompt: {prompt}\n")
    
    inference = MedicalLLMInference(model_name=model)
    result = inference.generate.remote(prompt, debug=debug)
    
    print(f"\n{'='*60}")
    print(f"Response:\n{result['text']}")
    print(f"{'='*60}")
    print(f"\nMetadata:")
    print(f"  - Template used: {result.get('template_used', 'unknown')}")
    print(f"  - Prompt tokens: {result.get('prompt_tokens')}")
    print(f"  - Completion tokens: {result.get('completion_tokens')}")
    print(f"  - Finish reason: {result.get('finish_reason')}")
