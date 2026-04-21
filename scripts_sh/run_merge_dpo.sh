#!/bin/bash
#SBATCH --job-name=Merge_DPO
#SBATCH --output=../output/logs/merge_dpo-%j.out
#SBATCH --error=../output/logs/merge_dpo-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --account=cpca183702024

# --- CONFIGURAÇÃO ---
PROJECT_ROOT="/users5/cpca183702024/finance1/bernardoleite"
SIF_IMAGE="$PROJECT_ROOT/common/images/llm_final.sif"
HF_CACHE="$PROJECT_ROOT/common/hf_cache"
SRC_DIR="$PROJECT_ROOT/projects/projeto_final_LLM/src"

# Variáveis de Ambiente
export HF_HOME=$HF_CACHE
export TORCH_HOME=$HF_CACHE
export APPTAINER_BIND="$PROJECT_ROOT:$PROJECT_ROOT"

# Modo Offline
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "--- Job Merge DPO Iniciado em $(date) ---"
cd $SRC_DIR

# Executa o script de merge universal
apptainer exec --nv \
    --bind $PROJECT_ROOT:$PROJECT_ROOT \
    $SIF_IMAGE \
    python3 merge_dpo_universal.py

echo "--- Job Merge DPO Finalizado em $(date) ---"
