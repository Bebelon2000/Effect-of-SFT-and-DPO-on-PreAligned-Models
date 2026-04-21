#!/bin/bash
#SBATCH --job-name=Gen_4k
#SBATCH --output=../logs/gen_4k_%j.out
#SBATCH --error=../logs/gen_4k_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8       # Aumentei de 4 para 8 (ajuda no data loader)
#SBATCH --gres=gpu:a100:1       # ESPECIFICIDADE: Pedir A100 explicitamente
#SBATCH --mem=64G
#SBATCH --time=06:00:00         
#SBATCH --partition=gpu
#SBATCH --account=cpca183702024

export HF_HOME="/users5/cpca183702024/finance1/bernardoleite/common/hf_cache"
export HF_HUB_OFFLINE=1

# Bind mount para garantir acesso
apptainer exec --nv \
    --bind /users5/cpca183702024/finance1/bernardoleite:/users5/cpca183702024/finance1/bernardoleite \
    /users5/cpca183702024/finance1/bernardoleite/common/images/llm_final.sif \
    python3 ../src/generate_4k_responses.py
