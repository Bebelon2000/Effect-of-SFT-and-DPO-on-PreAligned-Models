import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import json
import os

# --- CONFIGURAÇÃO ---
BASE_PATH = "/users5/cpca183702024/finance1/bernardoleite"
OUTPUT_FILE = f"{BASE_PATH}/projects/projeto_final_LLM/output/responses_MINI_test.json"
SEED = 42

# Caminhos (Confirme se estão corretos no seu sistema)
MODELS = {
    "Base": f"{BASE_PATH}/common/hf_cache/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
    "SFT": f"{BASE_PATH}/projects/projeto_final_LLM/output/Llama-8B-SFT-Merged",
    "Just_DPO": f"{BASE_PATH}/projects/projeto_final_LLM/output/Llama-3-Just-DPO-Final",
    "SFT_DPO": f"{BASE_PATH}/projects/projeto_final_LLM/output/Llama-3-SFT-DPO-Final"
}

DATASET_PATH = f"{BASE_PATH}/common/hf_cache/Anthropic___hh-rlhf/default/0.0.0/09be8c5bbc57cb3887f3a9732ad6aa7ec602a1fa/hh-rlhf-test.arrow"

def load_mini_prompts():
    print("--- Carregando 5 Prompts Aleatórios ---")
    try:
        ds = load_dataset("arrow", data_files={"test": DATASET_PATH}, split="test")
        ds = ds.shuffle(seed=SEED).select(range(5)) # Seleciona apenas 5
        
        prompts = []
        for item in ds:
            full_text = item['chosen']
            # Lógica de limpeza do prompt
            if "\n\nAssistant:" in full_text:
                prompt_clean = full_text.split("\n\nAssistant:")[0].replace("\n\nHuman:", "").strip()
            else:
                prompt_clean = full_text
            prompts.append(prompt_clean)
        return prompts
    except Exception as e:
        print(f"ERRO ao carregar dataset: {e}")
        return []

def generate_mini(model_name, model_path, prompts):
    print(f"\n>>> Testando Modelo: {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # --- CORREÇÃO CRÍTICA DE PADDING ---
        tokenizer.padding_side = "left" 
        
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.bfloat16, 
            device_map="auto",
            attn_implementation="flash_attention_2"
        )
    except Exception as e:
        print(f"Erro ao carregar modelo {model_name}: {e}")
        return ["ERRO"] * len(prompts)

    responses = []
    
    # Processa um a um para vermos o print no log
    for p in prompts:
        messages = [{"role": "user", "content": p}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = tokenizer(input_text, return_tensors="pt", padding=True).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=100, 
                do_sample=True, 
                temperature=0.6,
                pad_token_id=tokenizer.pad_token_id
            )
            
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Lógica de corte da resposta
        if "assistant" in decoded:
            resp = decoded.split("assistant")[-1].strip()
        elif "Assistant:" in decoded:
            resp = decoded.split("Assistant:")[-1].strip()
        else:
            resp = decoded.replace(input_text, "").strip() # Fallback
            
        responses.append(resp)
        print(f"   [Prompt]: {p[:40]}...")
        print(f"   [Resp]: {resp[:60]}...") # Mostra o início da resposta no log
        
    del model
    del tokenizer
    torch.cuda.empty_cache()
    return responses

if __name__ == "__main__":
    prompts = load_mini_prompts()
    
    if not prompts:
        print("Sem prompts para processar. Saindo.")
        exit(1)

    final_data = [{"prompt": p, "responses": {}} for p in prompts]

    for name, path in MODELS.items():
        if os.path.exists(path) or "models--" in path:
            resps = generate_mini(name, path, prompts)
            for i, r in enumerate(resps):
                final_data[i]["responses"][name] = r
        else:
            print(f"Caminho inválido para {name}: {path}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final_data, f, indent=2, ensure_ascii=False)
    print(f"\nSUCESSO! Teste concluído. Verifique: {OUTPUT_FILE}")