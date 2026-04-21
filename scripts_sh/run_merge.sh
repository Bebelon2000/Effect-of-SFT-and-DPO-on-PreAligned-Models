#!/bin/bash
#SBATCH --job-name=Llama_Merge
#SBATCH --output=../output/logs/merge-%j.out
#SBATCH --error=../output/logs/merge-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=cpca183702024
## #SBATCH --qos=gpu183702024

# --- CONFIGURAÇÃO ---
PROJECT_ROOT="/users5/cpca183702024/finance1/bernardoleite"
SIF_IMAGE="$PROJECT_ROOT/common/images/llm_final.sif"  # Usando a imagem full que tem dados
SRC_DIR="$PROJECT_ROOT/projects/projeto_final_LLM/src"

# Variáveis de Cache e Offline
export HF_HOME="/data/hf_cache"
export HF_DATASETS_CACHE="/data/hf_cache"
export HF_HUB_OFFLINE=1
export HF_TOKEN="secret-token" # Token por segurança

echo "--- Merge Iniciado ---"
cd $SRC_DIR

apptainer exec --nv \
    --bind $PROJECT_ROOT:$PROJECT_ROOT \
    $SIF_IMAGE \
    python3 -u merge_model.py

echo "--- Merge Finalizado ---"
