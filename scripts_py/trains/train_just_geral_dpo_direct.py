import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from trl import DPOTrainer, DPOConfig
import os

def main():
    # --- CONFIGURAÇÕES ---
    
    # 1. MODELO BASE (O Llama Original da Meta)
    # Usamos o caminho absoluto do snapshot para funcionar OFFLINE
    MODEL_PATH = "/users5/cpca183702024/finance1/bernardoleite/common/hf_cache/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
    
    # 2. OUTPUT DIFERENTE (Para não misturar com o futuro SFT+DPO)
    OUTPUT_DIR = "../output/final_model_dpo_direct"
    
    DATASET_NAME = "Anthropic/hh-rlhf"

    print(f"--- Iniciando Treino DPO (Direct on Base) ---")
    print(f"Modelo Base: {MODEL_PATH}")

    # Configuração de Quantização (QLoRA 4-bit)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Carregar Modelo Base
    print("Carregando Modelo Base...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        use_cache=False,
        local_files_only=True 
    )
    
    model = prepare_model_for_kbit_training(model)

    # Carregar Tokenizer
    print("Carregando Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Carregar Dataset
    print(f"Carregando Dataset: {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME, split="train")

    # Configuração LoRA
    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=['k_proj', 'q_proj', 'v_proj', 'o_proj', "gate_proj", "down_proj", "up_proj"]
    )

    # Argumentos de Treino
    training_args = DPOConfig(
        output_dir=OUTPUT_DIR,
        beta=0.1,
        learning_rate=5e-7,           # LR conservadora
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        optim="paged_adamw_32bit",
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        bf16=True,
        report_to="none",
        remove_unused_columns=False,
        max_length=2048,
        max_prompt_length=1024,
        warmup_steps=100,
        lr_scheduler_type="cosine",
    )

    # Inicializar DPOTrainer
    print("Inicializando DPOTrainer...")
    trainer = DPOTrainer(
        model=model,
        ref_model=None, # O trainer cria a cópia implícita da Base como referência
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    # Treinar
    print("--- Começando Treino DPO Direto ---")
    trainer.train()
    
    # Salvar
    print(f"Salvando Modelo Final em {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    print("--- DPO Finalizado ---")

if __name__ == "__main__":
    main()
