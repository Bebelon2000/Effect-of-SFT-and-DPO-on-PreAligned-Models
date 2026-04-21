import torch
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from trl import DPOTrainer, DPOConfig
import os

def main():
    # --- CONFIGURAÇÕES ---
    # AQUI ESTÁ A CHAVE: Apontar para o modelo SFT que fundimos antes
    MODEL_PATH = "../output/Llama-8B-SFT-Merged"
    
    # Pasta de saída exclusiva para este experimento
    OUTPUT_DIR = "../output/final_model_dpo_sft"
    
    DATASET_NAME = "Anthropic/hh-rlhf"

    print(f"--- Iniciando Treino DPO (Sobre SFT) ---")
    print(f"Modelo Base: {MODEL_PATH}")

    # 1. Quantização (4-bit para economizar memória)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # 2. Carregar Modelo (Offline e Local)
    print("Carregando Modelo SFT...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        use_cache=False,
        local_files_only=True
    )
    
    model = prepare_model_for_kbit_training(model)

    # 3. Carregar Tokenizer
    print("Carregando Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    
    # Fix para o Pad Token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4. Dataset
    print(f"Carregando Dataset: {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME, split="train")

    # 5. Configuração LoRA (Novos adaptadores para o DPO)
    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=['k_proj', 'q_proj', 'v_proj', 'o_proj', "gate_proj", "down_proj", "up_proj"]
    )

    # 6. Argumentos de Treino
    training_args = DPOConfig(
        output_dir=OUTPUT_DIR,
        beta=0.1,
        learning_rate=5e-7,           # LR baixa e segura
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        optim="paged_adamw_32bit",
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,           # Guarda apenas os 3 últimos checkpoints
        bf16=True,
        report_to="none",
        remove_unused_columns=False,
        max_length=2048,
        max_prompt_length=1024,
        warmup_steps=100,
        lr_scheduler_type="cosine",
    )

    # 7. Treinar
    print("Inicializando DPOTrainer...")
    trainer = DPOTrainer(
        model=model,
        ref_model=None, # TRL cria a referência implicitamente
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    print("--- Começando Treino DPO SFT ---")
    trainer.train()
    
    print(f"Salvando Modelo Final em {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    print("--- DPO SFT Finalizado ---")

if __name__ == "__main__":
    main()
