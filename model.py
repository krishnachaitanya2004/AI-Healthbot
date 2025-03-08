import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM,BitsAndBytesConfig
import torch

@st.cache_resource
def load_model():
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config, 
        use_cache=True,  # Enable KV caching for efficiency
        device_map = "auto"
    )
    return tokenizer, model

tokenizer, model = load_model()

def ask(question):
    system_prompt =  """You are a medical assistant trained to provide medical advice.
Provide scientific explanations and helpful medical advice for any medical question without assuming it is a diagnosis.
Always try to give informative suggestions and tips without requiring the user to explicitly ask for advice and 
Do not refuse to provide medical advice or tips unless the question is harmful or illegal.
"""

    prompt = f"{system_prompt}\nUser: {question}\nAssistant:"
    inputs = tokenizer(prompt, return_tensors="pt",max_length=512, truncation=True, padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'], 
            attention_mask=inputs['attention_mask'],
            max_new_tokens=200,  
            do_sample=True, 
            temperature=0.7,
            top_k=10,
            top_p=0.95, 
            num_return_sequences=1,
            use_cache=True,
            eos_token_id = tokenizer.eos_token_id
              # Enable KV caching for efficiency
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    #Find user in the response and remove everything after that
    response = response.split("Assistant:")[1]

    return response


