#!/bin/bash
#SBATCH --job-name=Llama_SFT
#SBATCH --output=../output/logs/slurm-%j.out
#SBATCH --error=../output/logs/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --account=cpca183702024
## #SBATCH --qos=gpu183702024

# --- CONFIGURAÇÃO DE AMBIENTE ---
PROJECT_ROOT="/users5/cpca183702024/finance1/bernardoleite"
# --- AQUI ESTÁ A MUDANÇA: Usar a imagem FINAL ---
SIF_IMAGE="$PROJECT_ROOT/common/images/llm_final.sif"
HF_CACHE="$PROJECT_ROOT/common/hf_cache"
SRC_DIR="$PROJECT_ROOT/projects/projeto_final_LLM/src"

# Token
export HF_TOKEN="secret_token"

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Cache
export HF_HOME=$HF_CACHE
export HF_DATASETS_CACHE=$HF_CACHE
export TORCH_HOME=$HF_CACHE
export APPTAINER_BIND="$PROJECT_ROOT:$PROJECT_ROOT"

echo "--- Job SFT Iniciado em $(date) ---"
echo "Nó: $HOSTNAME"

cd $SRC_DIR

# Execução (Sem PYTHONPATH extra!)
apptainer exec --nv \
    --bind $PROJECT_ROOT:$PROJECT_ROOT \
    $SIF_IMAGE \
    python3 -u train_sft.py

echo "--- Job Finalizado em $(date) ---"
