import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
import os
import glob

# --- CONFIGURAÇÃO DE CAMINHOS ABSOLUTOS ---
BASE_PATH = "/users5/cpca183702024/finance1/bernardoleite"

# Função de Auto-Discovery do Modelo Base
def find_base_model_path():
    possible_paths = [
        f"{BASE_PATH}/common/hf_cache/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/*",
        f"{BASE_PATH}/common/hf_cache/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/*"
    ]
    for pattern in possible_paths:
        found = glob.glob(pattern)
        for p in found:
            if os.path.exists(os.path.join(p, "config.json")):
                print(f"[AUTO] Modelo Base encontrado: {p}")
                return p
    print("[AVISO] Base não encontrado automaticamente. Usando fallback.")
    return f"{BASE_PATH}/common/hf_cache/models--meta-llama--Meta-Llama-3.1-8B-Instruct"

BASE_MODEL_PATH = find_base_model_path()

MODELS_PATHS = {
    "Base": BASE_MODEL_PATH,
    "SFT": f"{BASE_PATH}/projects/projeto_final_LLM/output/Llama-8B-SFT-Merged",
    "Just_DPO": f"{BASE_PATH}/projects/projeto_final_LLM/output/Llama-3-Just-DPO-Final",
    "SFT_DPO": f"{BASE_PATH}/projects/projeto_final_LLM/output/Llama-3-SFT-DPO-Final"
}

TEST_ARROW_FILE = f"{BASE_PATH}/common/hf_cache/Anthropic___hh-rlhf/default/0.0.0/09be8c5bbc57cb3887f3a9732ad6aa7ec602a1fa/hh-rlhf-test.arrow"

def get_batch_logps(model, tokenizer, batch):
    """
    Calcula log-probs com PROMPT MASKING e LENGTH NORMALIZATION.
    Técnica SOTA: Ignora o prompt comum e normaliza apenas pelos tokens da resposta.
    """
    chosen_texts = batch['chosen']
    rejected_texts = batch['rejected']
    
    # Tokenização com padding à direita
    inputs_c = tokenizer(chosen_texts, return_tensors='pt', padding=True, truncation=True, max_length=2048).to(model.device)
    inputs_r = tokenizer(rejected_texts, return_tensors='pt', padding=True, truncation=True, max_length=2048).to(model.device)
    
    with torch.no_grad():
        logits_c = model(**inputs_c).logits
        logits_r = model(**inputs_r).logits
    
    labels_c = inputs_c.input_ids.clone()
    labels_r = inputs_r.input_ids.clone()
    
    # 1. Mascarar Padding original (-100)
    labels_c[labels_c == tokenizer.pad_token_id] = -100
    labels_r[labels_r == tokenizer.pad_token_id] = -100

    # 2. PROMPT MASKING (A grande melhoria)
    # Descobre onde o Prompt acaba comparando Chosen vs Rejected
    # Onde eles são iguais, é Prompt. Onde divergem, é Resposta.
    # Cria uma máscara booleana onde os tokens são diferentes
    # (Adicionamos padding dummy para alinhar tamanhos se necessário, mas aqui tratamos inputs separados)
    
    # Nota: Para fazer masking vetorizado eficiente sem alinhar tensores C e R (que têm tamanhos diferentes),
    # calculamos a loss total e depois subtraímos a loss do prefixo comum? 
    # Mais robusto: Vamos iterar no batch para encontrar o prefixo exato. 
    # Como o batch é pequeno (8), o custo de CPU é negligenciável comparado ao forward pass.
    
    for i in range(len(chosen_texts)):
        # Encontra o índice onde o token difere
        # Simples: iterar até encontrar diferença
        # (Na prática, mascaramos a loss dos tokens anteriores a esse índice)
        
        # Truque: Usamos a loss não reduzida. Vamos apenas mascarar os labels.
        # Precisamos garantir que não mascaramos tudo se forem iguais (impossível no dataset)
        pass 
        # A lógica vetorizada abaixo ("Divergence Finding") é mais rápida que loops python
        
    # Lógica Vetorizada de Máscara de Prompt:
    # Como inputs_c e inputs_r podem ter tamanhos diferentes devido ao padding,
    # truncamos para o tamanho mínimo para comparar o prefixo.
    min_len = min(inputs_c.input_ids.size(1), inputs_r.input_ids.size(1))
    
    # Comparação: Onde são iguais?
    # diff_mask = 1 onde são diferentes, 0 onde são iguais
    common_prefix = (inputs_c.input_ids[:, :min_len] == inputs_r.input_ids[:, :min_len])
    
    # Encontrar o primeiro índice de divergência para cada linha
    # (argmax retorna o primeiro True da negação, ou seja, o primeiro diferente)
    divergence_idx = (~common_prefix).int().argmax(dim=1)
    
    # Aplicar a máscara aos LABELS
    for i in range(len(chosen_texts)):
        idx = divergence_idx[i]
        # Se idx for 0, significa que divergem logo no início (raro, mas possível)
        if idx > 0:
            labels_c[i, :idx] = -100
            labels_r[i, :idx] = -100

    # Shift para alinhar previsão (Token T prevê T+1)
    shift_logits_c = logits_c[..., :-1, :].contiguous()
    shift_labels_c = labels_c[..., 1:].contiguous()
    shift_logits_r = logits_r[..., :-1, :].contiguous()
    shift_labels_r = labels_r[..., 1:].contiguous()

    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    
    # Loss por token (raw)
    token_loss_c = loss_fct(shift_logits_c.transpose(1, 2), shift_labels_c)
    token_loss_r = loss_fct(shift_logits_r.transpose(1, 2), shift_labels_r)
    
    # Máscara de tokens válidos (que não são padding NEM prompt)
    mask_c = (shift_labels_c != -100).float()
    mask_r = (shift_labels_r != -100).float()
    
    # Soma da loss válida
    sum_loss_c = (token_loss_c * mask_c).sum(dim=1)
    sum_loss_r = (token_loss_r * mask_r).sum(dim=1)
    
    # 3. LENGTH NORMALIZATION (Divisão pelo tamanho real da resposta)
    # Adicionamos 1e-9 para evitar divisão por zero
    avg_loss_c = sum_loss_c / (mask_c.sum(dim=1) + 1e-9)
    avg_loss_r = sum_loss_r / (mask_r.sum(dim=1) + 1e-9)
    
    return -avg_loss_c, -avg_loss_r

def evaluate_model(name, path, dataset):
    print(f"\n--- Avaliando Modelo: {name} (SOTA Mode) ---")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(path)
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="flash_attention_2"
        )
    except Exception as e:
        print(f"ERRO: {e}")
        return 0.0

    model.eval()
    correct = 0
    total = 0
    batch_size = 8
    
    for i in tqdm(range(0, len(dataset), batch_size), desc=f"Eval {name}"):
        batch = dataset[i:i+batch_size]
        log_probs_c, log_probs_r = get_batch_logps(model, tokenizer, batch)
        
        preds = (log_probs_c > log_probs_r).int()
        correct += preds.sum().item()
        total += preds.size(0)
        
    acc = correct / total
    print(f" >> Acurácia Normalizada {name}: {acc:.4f}")
    
    del model
    del tokenizer
    torch.cuda.empty_cache()
    return acc

if __name__ == "__main__":
    print("Iniciando Benchmark SOTA (Masked + Normalized)...")
    try:
        if os.path.exists(TEST_ARROW_FILE):
            dataset = load_dataset("arrow", data_files={"test": TEST_ARROW_FILE}, split="test")
            print(f"Dataset carregado: {len(dataset)} linhas.")
        else:
            print("ERRO: Dataset não encontrado.")
            exit(1)
    except Exception as e:
        print(f"Erro Fatal: {e}")
        exit(1)

    results = []
    for name, path in MODELS_PATHS.items():
        if os.path.exists(path) or glob.glob(path):
            acc = evaluate_model(name, path, dataset)
            results.append({"Modelo": name, "Acuracia": acc})
        else:
            print(f"AVISO: {path} não encontrado.")
    
    df = pd.DataFrame(results)
    print("\n=== RESULTADOS FINAIS (NORMALIZADOS) ===")
    print(df)
    df.to_csv(f"{BASE_PATH}/projects/projeto_final_LLM/output/benchmark_hhrlhf_8k_normalized.csv", index=False)
