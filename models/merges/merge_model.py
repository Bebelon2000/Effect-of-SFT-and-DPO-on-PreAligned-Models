import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

# --- CONFIGURAÇÕES ---
# Caminhos baseados na execução a partir de 'src'
BASE_MODEL_ID = "/users5/cpca183702024/finance1/bernardoleite/common/hf_cache/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"  # O nome para buscar no cache
ADAPTER_PATH = "../output/final_model_sft"               # Onde salvamos o SFT
OUTPUT_PATH = "../output/Llama-8B-SFT-Merged"             # Onde vai ficar o modelo fundido

print(f"--- Iniciando Merge ---")
print(f"Base: {BASE_MODEL_ID}")
print(f"Adapter: {ADAPTER_PATH}")

# 1. Carregar Modelo Base (Em FP16 para não estourar memória no merge)
print("Carregando Modelo Base...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
    local_files_only=True,
)

# 2. Carregar Tokenizer
print("Carregando Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

# 3. Carregar Adaptadores LoRA
print("Carregando LoRA Adapters...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

# 4. Fazer o Merge (Fundir pesos)
print("Fundindo pesos (Merge & Unload)...")
model = model.merge_and_unload()

# 5. Salvar o Modelo Completo
print(f"Salvando modelo fundido em: {OUTPUT_PATH}")
model.save_pretrained(OUTPUT_PATH)
tokenizer.save_pretrained(OUTPUT_PATH)

print("--- Sucesso! Modelo pronto para DPO. ---")
