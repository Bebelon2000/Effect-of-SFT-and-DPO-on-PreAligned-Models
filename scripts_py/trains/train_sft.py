import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
import re

# --- CONFIGURAÇÕES ---
# O script assume que rodamos na pasta 'src', então 'output' está um nível acima
MODEL_NAME = "/users5/cpca183702024/finance1/bernardoleite/common/hf_cache/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
DATASET_NAME = "Anthropic/hh-rlhf"

OUTPUT_DIR = "../output/checkpoints"
FINAL_MODEL_DIR = "../output/final_model_sft"

# --- CONFIGURAÇÃO QLoRA (4-bit) ---
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

def format_prompts(batch):
    """
    Formata o dataset HH-RLHF para o template oficial do Llama 3.
    """
    outputs = []
    for full_text in batch['chosen']:
        try:
            # Divide o texto entre Human e Assistant
            parts = re.split(r'(\n\nHuman: |\n\nAssistant: )', str(full_text))
            if not parts[0]: parts = parts[1:]
            
            msgs = []
            i = 0
            while i < len(parts) - 1:
                role = "user" if "Human" in parts[i] else "assistant"
                content = parts[i+1].strip()
                msgs.append({"role": role, "content": content})
                i += 2
            
            # Aplica as tags especiais do Llama 3
            formatted = ""
            for m in msgs:
                formatted += f"<|start_header_id|>{m['role']}<|end_header_id|>\n\n{m['content']}<|eot_id|>"
            
            # Adiciona o prompt para o assistente começar a gerar
            if msgs and msgs[-1]['role'] == 'user':
                formatted += "<|start_header_id|>assistant<|end_header_id|>\n\n"

            outputs.append(formatted)
        except Exception:
            outputs.append(None)
    return {"text": outputs}

def main():
    print(f"--- Iniciando Treino SFT ---")
    
    # Verificações de Hardware
    print(f"CUDA Disponível: {torch.cuda.is_available()}")
    n_gpus = torch.cuda.device_count()
    print(f"Número de GPUs: {n_gpus}")
    
    # 1. Carregar Tokenizer
    print("Carregando Tokenizer do cache...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 2. Carregar Modelo
    print("Carregando Modelo do cache...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="sdpa" # Flash Attention Otimizado
    )
    model = prepare_model_for_kbit_training(model)

    # 3. Configurar LoRA
    peft_config = LoraConfig(
        r=16, 
        lora_alpha=16, 
        lora_dropout=0.05, 
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # 4. Dataset
    print("Carregando Dataset...")
    dataset = load_dataset(DATASET_NAME, split="train")
    
    print("Formatando Dataset...")
    dataset = dataset.map(format_prompts, batched=True, num_proc=4)
    dataset = dataset.filter(lambda x: x["text"] is not None)

	# 5. Argumentos de Treino (SFTConfig)
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        dataset_text_field="text",
        packing=True,                 # Definido AQUI
        max_length=2048,          # Definido AQUI
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=5,
        optim="paged_adamw_32bit",
        report_to="none",
        warmup_steps=100,             # Corrigido (era ratio)
        lr_scheduler_type="constant",
        gradient_checkpointing=True,  # ADICIONADO: Essencial para memória
        gradient_checkpointing_kwargs={'use_reentrant': False} # ADICIONADO: Estabilidade
    )

    # 6. Inicializar Trainer
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_args
        # REMOVIDO: max_seq_length (já está no args)
        # REMOVIDO: packing (já está no args)
    )

    print("--- Trainer Inicializado. Começando treino... ---")
    trainer.train()

    print(f"--- Treino Finalizado. Salvando em {FINAL_MODEL_DIR} ---")
    trainer.save_model(FINAL_MODEL_DIR)
    tokenizer.save_pretrained(FINAL_MODEL_DIR)

if __name__ == "__main__":
    main()

