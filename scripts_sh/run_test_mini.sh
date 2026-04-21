#!/bin/bash
#SBATCH --job-name=Test_Mini
#SBATCH --output=../logs/test_mini_%j.out
#SBATCH --error=../logs/test_mini_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1       
#SBATCH --mem=32G
#SBATCH --time=00:15:00         # <--- 15 min garante prioridade máxima (backfill)
#SBATCH --partition=gpu         # <--- Mantemos GPU para compatibilidade de hardware
#SBATCH --account=cpca183702024

export HF_HOME="/users5/cpca183702024/finance1/bernardoleite/common/hf_cache"
export HF_HUB_OFFLINE=1

echo "--- Iniciando Mini Teste (5 Prompts) com Padding Fix ---"

apptainer exec --nv \
    --bind /users5/cpca183702024/finance1/bernardoleite:/users5/cpca183702024/finance1/bernardoleite \
    /users5/cpca183702024/finance1/bernardoleite/common/images/llm_final.sif \
    python3 ../src/test_gen_just_5_mini.py
