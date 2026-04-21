import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import os

# --- CONFIGURAÇÃO ---
# Caminho do modelo MERGED (Verifique se o nome da pasta é exatamente este)
# Se o seu merge criou "Llama-3-SFT-Merged" ou "Llama-8B-...", ajuste aqui:
MODEL_PATH = "../output/Llama-8B-SFT-Merged"

print(f"--- Carregando Modelo SFT Local: {MODEL_PATH} ---")

try:
    # 1. Carregar Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    
    # 2. Carregar Modelo
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.float16,
        local_files_only=True
    )
    print("--- Modelo Carregado com Sucesso! (OFFLINE) ---")
except Exception as e:
    print(f"\nERRO CRÍTICO: Não encontrei a pasta. Verifique se o nome em MODEL_PATH está igual à pasta em ../output/")
    print(f"Erro detalhado: {e}")
    sys.exit()

print("--- Pode começar a conversar (Escreva 'sair' para fechar) ---")

while True:
    try:
        user_input = input("\nVocê: ")
        if user_input.lower() in ["sair", "exit", "quit"]:
            break
        
        # Template do Llama 3
        messages = [{"role": "user", "content": user_input}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        outputs = model.generate(
            **inputs, 
            max_new_tokens=200, 
            do_sample=True, 
            temperature=0.7,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )
        
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"Llama SFT: {response}")
        
    except KeyboardInterrupt:
        break
