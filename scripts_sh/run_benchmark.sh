#!/bin/bash
#SBATCH --job-name=Llama_Bench
#SBATCH --output=../output/logs/bench-%j.out
#SBATCH --error=../output/logs/bench-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=cpca183702024

# --- CONFIGURAÇÃO ---
PROJECT_ROOT="/users5/cpca183702024/finance1/bernardoleite"
SIF_IMAGE="$PROJECT_ROOT/common/images/llm_final.sif"
HF_CACHE="$PROJECT_ROOT/common/hf_cache"
SRC_DIR="$PROJECT_ROOT/projects/projeto_final_LLM/src"

# Variáveis de Ambiente para Offline Mode
export HF_HOME=$HF_CACHE
export TORCH_HOME=$HF_CACHE
export APPTAINER_BIND="$PROJECT_ROOT:$PROJECT_ROOT"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "--- Benchmark Iniciado em $(date) ---"
cd $SRC_DIR

# Executa o script Python
apptainer exec --nv \
    --bind $PROJECT_ROOT:$PROJECT_ROOT \
    $SIF_IMAGE \
    python3 benchmark_master.py

echo "--- Benchmark Finalizado em $(date) ---"
