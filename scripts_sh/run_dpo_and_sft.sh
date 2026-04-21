#!/bin/bash
#SBATCH --job-name=Llama_DPO_SFT
#SBATCH --output=../output/logs/dpo_sft-%j.out
#SBATCH --error=../output/logs/dpo_sft-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --account=cpca183702024

# --- CONFIGURAÇÃO ---
PROJECT_ROOT="/users5/cpca183702024/finance1/bernardoleite"
SIF_IMAGE="$PROJECT_ROOT/common/images/llm_final.sif"
HF_CACHE="$PROJECT_ROOT/common/hf_cache"
SRC_DIR="$PROJECT_ROOT/projects/projeto_final_LLM/src"

export HF_HOME=$HF_CACHE
export HF_DATASETS_CACHE=$HF_CACHE
export TORCH_HOME=$HF_CACHE
export APPTAINER_BIND="$PROJECT_ROOT:$PROJECT_ROOT"

# Token e Offline Mode
export HF_TOKEN="secret-token"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "--- Job DPO SFT Iniciado ---"
cd $SRC_DIR

apptainer exec --nv \
    --bind $PROJECT_ROOT:$PROJECT_ROOT \
    $SIF_IMAGE \
    python3 -u train_dpo_and_sft.py

echo "--- Job DPO SFT Finalizado ---"
