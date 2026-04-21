import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import json
import os

# --- CONFIGURAÇÃO DE CAMINHOS ABSOLUTOS ---
BASE_PATH = "/users5/cpca183702024/finance1/bernardoleite"
OUTPUT_FILE = f"{BASE_PATH}/projects/projeto_final_LLM/output/responses_4k_dataset.json"
SEED = 42

# Caminhos dos Modelos
MODELS = {
    "Base": f"{BASE_PATH}/common/hf_cache/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
    "SFT": f"{BASE_PATH}/projects/projeto_final_LLM/output/Llama-8B-SFT-Merged",
    "Just_DPO": f"{BASE_PATH}/projects/projeto_final_LLM/output/Llama-3-Just-DPO-Final",
    "SFT_DPO": f"{BASE_PATH}/projects/projeto_final_LLM/output/Llama-3-SFT-DPO-Final"
}

DATASET_PATH = f"{BASE_PATH}/common/hf_cache/Anthropic___hh-rlhf/default/0.0.0/09be8c5bbc57cb3887f3a9732ad6aa7ec602a1fa/hh-rlhf-test.arrow"

def load_and_prep_dataset(n_samples=4000):
    print(f"--- Carregando {n_samples} Prompts Aleatórios ---")
    try:
        ds = load_dataset("arrow", data_files={"test": DATASET_PATH}, split="test")
        ds = ds.shuffle(seed=SEED).select(range(n_samples))
        
        prompts = []
        for item in ds:
            full_text = item['chosen']
            if "\n\nAssistant:" in full_text:
                prompt_clean = full_text.split("\n\nAssistant:")[0].replace("\n\nHuman:", "").strip()
            else:
                prompt_clean = full_text
            prompts.append(prompt_clean)
        return prompts
    except Exception as e:
        print(f"ERRO CRÍTICO ao carregar dataset: {e}")
        exit(1)

def generate_responses(model_name, model_path, prompts):
    print(f"\n>>> Gerando 4k Respostas para: {model_name}...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # --- CORREÇÃO DE PADDING (CRUCIAL) ---
        tokenizer.padding_side = "left" 
        
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2"
        )
    except Exception as e:
        print(f"ERRO ao carregar {model_name}: {e}")
        return ["ERRO_LOAD"] * len(prompts)
        
    model.eval()
    responses = []
    batch_size = 16  # Eficiente para A100
    
    for i in tqdm(range(0, len(prompts), batch_size), desc=f"Inferência {model_name}"):
        batch_prompts = prompts[i:i+batch_size]
        
        # Aplica Chat Template
        formatted_batch = []
        for p in batch_prompts:
            messages = [{"role": "user", "content": p}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            formatted_batch.append(text)
            
        inputs = tokenizer(formatted_batch, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id
            )
            
        # Decodificação limpa
        for j, output in enumerate(outputs):
            decoded = tokenizer.decode(output, skip_special_tokens=True)
            
            if "assistant" in decoded:
                resp = decoded.split("assistant")[-1].strip()
            elif "Assistant:" in decoded:
                resp = decoded.split("Assistant:")[-1].strip()
            else:
                resp = decoded.replace(batch_prompts[j], "").strip()
                
            responses.append(resp)
            
    del model
    del tokenizer
    torch.cuda.empty_cache()
    return responses

if __name__ == "__main__":
    # 1. Carregar Prompts
    prompts = load_and_prep_dataset(4000)
    
    # 2. Inicializar Estrutura de Dados
    # Se o arquivo já existir, poderíamos carregar para continuar, mas vamos simplificar e sobrescrever
    final_data = [{"id": i, "prompt": p, "responses": {}} for i, p in enumerate(prompts)]

    # 3. Loop pelos Modelos
    for name, path in MODELS.items():
        if os.path.exists(path) or "models--" in path:
            model_responses = generate_responses(name, path, prompts)
            
            # Guardar na estrutura principal
            for i, resp in enumerate(model_responses):
                final_data[i]["responses"][name] = resp
            
            # --- CHECKPOINT DE SEGURANÇA ---
            # Salva a cada modelo concluído. Se o job morrer no meio, já temos os anteriores.
            print(f"Salvando checkpoint após {name}...")
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(final_data, f, indent=2, ensure_ascii=False)
        else:
            print(f"AVISO: Caminho não encontrado: {path}")

    print(f"\n✅ CONCLUÍDO! 16.000 respostas geradas em: {OUTPUT_FILE}")
