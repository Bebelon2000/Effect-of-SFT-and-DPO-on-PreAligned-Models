import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
import os

# --- CONFIGURAÇÃO DE CAMINHOS ---
BASE_PATH = "/users5/cpca183702024/finance1/bernardoleite"

MODELS_PATHS = {
    # Modelo Base (Snapshot)
    "Base": f"{BASE_PATH}/common/hf_cache/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
    # Seus Modelos
    "SFT": f"{BASE_PATH}/projects/projeto_final_LLM/output/Llama-8B-SFT-Merged",
    "Just_DPO": f"{BASE_PATH}/projects/projeto_final_LLM/output/Llama-3-Just-DPO-Final",
    "SFT_DPO": f"{BASE_PATH}/projects/projeto_final_LLM/output/Llama-3-SFT-DPO-Final"
}

# Arquivo Físico
TEST_ARROW_FILE = f"{BASE_PATH}/common/hf_cache/Anthropic___hh-rlhf/default/0.0.0/09be8c5bbc57cb3887f3a9732ad6aa7ec602a1fa/hh-rlhf-test.arrow"

def get_batch_logps(model, tokenizer, batch):
    """Calcula log-probs (Simplificado para teste)"""
    chosen_texts = [q + r for q, r in zip(batch['prompt'], batch['chosen'])]
    rejected_texts = [q + r for q, r in zip(batch['prompt'], batch['rejected'])]
    
    inputs_c = tokenizer(chosen_texts, return_tensors='pt', padding=True, truncation=True, max_length=1024).to(model.device)
    inputs_r = tokenizer(rejected_texts, return_tensors='pt', padding=True, truncation=True, max_length=1024).to(model.device)
    
    with torch.no_grad():
        logits_c = model(**inputs_c).logits
        logits_r = model(**inputs_r).logits
    
    # Lógica de Loss simplificada
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    labels_c = inputs_c.input_ids.clone()
    labels_c[labels_c == tokenizer.pad_token_id] = -100
    shift_logits_c = logits_c[..., :-1, :].contiguous()
    shift_labels_c = labels_c[..., 1:].contiguous()
    loss_c = loss_fct(shift_logits_c.transpose(1, 2), shift_labels_c).sum(dim=1)
    
    labels_r = inputs_r.input_ids.clone()
    labels_r[labels_r == tokenizer.pad_token_id] = -100
    shift_logits_r = logits_r[..., :-1, :].contiguous()
    shift_labels_r = labels_r[..., 1:].contiguous()
    loss_r = loss_fct(shift_logits_r.transpose(1, 2), shift_labels_r).sum(dim=1)
    
    return -loss_c, -loss_r

def evaluate_model(name, path, dataset):
    print(f"--- Testando {name} ---")
    try:
        tokenizer = AutoTokenizer.from_pretrained(path)
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map="auto")
        model.eval()
    except Exception as e:
        print(f"ERRO ao carregar {name}: {e}")
        return 0.0

    correct = 0
    # Loop simples
    for i in range(len(dataset)):
        batch = dataset[i:i+1] # Batch de 1 para teste
        lc, lr = get_batch_logps(model, tokenizer, batch)
        if lc > lr: correct += 1
        
    print(f" >> Resultado {name}: {correct}/{len(dataset)} acertos")
    return correct / len(dataset)

if __name__ == "__main__":
    print("=== MINI TESTE DE DEBUG (2 PROMPTS) ===")
    
    if os.path.exists(TEST_ARROW_FILE):
        full_dataset = load_dataset("arrow", data_files={"test": TEST_ARROW_FILE}, split="test")
        # --- AQUI ESTÁ O SEGREDO DO TESTE ---
        dataset = full_dataset.select(range(2)) 
        print(f"Dataset cortado para: {len(dataset)} linhas.")
    else:
        print("Dataset não encontrado!")
        exit(1)

    for name, path in MODELS_PATHS.items():
        if os.path.exists(path):
            evaluate_model(name, path, dataset)
        else:
            print(f"Modelo não encontrado: {path}")
            
    print("=== FIM DO TESTE ===")
