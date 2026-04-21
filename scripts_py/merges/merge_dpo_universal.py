import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import argparse

# Configuração para não tentar baixar coisas
os.environ["HF_HUB_OFFLINE"] = "1"

def merge_model(base_model_path, adapter_path, output_dir):
    print(f"--- Iniciando Merge ---")
    print(f"Base: {base_model_path}")
    print(f"Adapter: {adapter_path}")
    
    print("1. Carregando Modelo Base em FP16...")
    # Carregamos em float16 para facilitar o merge (não usar 4bit aqui para merge final de qualidade)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True
    )

    print("2. Carregando Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, local_files_only=True)

    print("3. Acoplando Adaptadores LoRA...")
    model = PeftModel.from_pretrained(base_model, adapter_path, local_files_only=True)

    print("4. Fundindo pesos (Merge & Unload)...")
    model = model.merge_and_unload()

    print(f"5. Salvando modelo completo em: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("--- Sucesso! Modelo Final Pronto. ---")

if __name__ == "__main__":
    # Vamos definir hardcoded para rodar um de cada vez ou criar um script bash inteligente
    # Exemplo para uso manual via edição ou argumentos
    
    # CASO 1: JUST DPO (Base Original + DPO Adapter)
    # Descomente abaixo para rodar o Just DPO
    # BASE = "/users5/cpca183702024/finance1/bernardoleite/common/hf_cache/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
    # ADAPTER = "../output/final_model_dpo_direct"
    # OUTPUT = "../output/Llama-3-Just-DPO-Final"
    
    # CASO 2: SFT + DPO (SFT Merged + DPO Adapter)
    # Descomente abaixo para rodar o SFT+DPO
    BASE = "../output/Llama-8B-SFT-Merged"
    ADAPTER = "../output/final_model_dpo_sft"
    OUTPUT = "../output/Llama-3-SFT-DPO-Final"
    
    merge_model(BASE, ADAPTER, OUTPUT)
