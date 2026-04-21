import json
import asyncio
import aiohttp
import os
import random
from tqdm.asyncio import tqdm

# --- CONFIGURAÇÃO DEEPINFRA ---
# 1. Obtenha a chave em: https://deepinfra.com/dash/api_keys
DEEPINFRA_API_KEY = "V7If2NkfFsMXrLTj8ALXTMV5ELYeqVwk"

# Configuração da DeepInfra (Compatível com OpenAI)
BASE_URL = "https://api.deepinfra.com/v1/openai/chat/completions"
# Nome exato do modelo na DeepInfra
MODEL_JUDGE = "meta-llama/Llama-3.3-70B-Instruct"

INPUT_FILE = "responses_4k_dataset.json"
PROMPT_FILE = "judge_prompt.txt"
OUTPUT_FILE = "evaluation_results_4k.jsonl"

# A DeepInfra aguenta muita carga. Pode por 10 ou 20.
CONCURRENCY_LIMIT = 15

def load_system_prompt():
    if not os.path.exists(PROMPT_FILE):
        print(f"ERRO: {PROMPT_FILE} não encontrado!")
        exit(1)
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        return f.read().strip()

async def evaluate_single_response(session, item, model_name, response_text, system_prompt, semaphore):
    async with semaphore: 
        prompt_content = f"User Prompt: {item['prompt']}\n\nAI Response: {response_text}"
        
        payload = {
            "model": MODEL_JUDGE,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_content}
            ],
            "temperature": 0.0,
            # DeepInfra suporta json_object para garantir o formato
            "response_format": {"type": "json_object"} 
        }

        headers = {
            "Authorization": f"Bearer {DEEPINFRA_API_KEY}",
            "Content-Type": "application/json"
        }

        # Tentativas Robustas
        attempts = 0
        while attempts < 10:
            attempts += 1
            try:
                async with session.post(BASE_URL, json=payload, headers=headers) as resp:
                    if resp.status == 200:
                        try:
                            result = await resp.json()
                            content = result['choices'][0]['message']['content']
                            eval_data = json.loads(content)
                            return {
                                "id": item['id'], "model": model_name,
                                "prompt": item['prompt'], "response": response_text,
                                "evaluation": eval_data
                            }
                        except:
                            # Se o JSON vier inválido, tenta de novo
                            continue
                    
                    elif resp.status == 429:
                        # Se der rate limit (raro se tiver crédito), espera um pouco
                        await asyncio.sleep(2)
                        continue
                    else:
                        # Erros de servidor
                        await asyncio.sleep(1)
                        continue
            except:
                await asyncio.sleep(1)
        return None

async def main():
    print(f"--- MODO DEEPINFRA (Velocidade Máxima) ---")
    
    system_prompt = load_system_prompt()
    if not os.path.exists(INPUT_FILE): return
    with open(INPUT_FILE, "r", encoding="utf-8") as f: data = json.load(f)

    # Verifica o que já foi feito (mantém o progresso da Groq!)
    processed_keys = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    processed_keys.add(f"{record['id']}_{record['model']}")
                except: pass
    
    tasks_to_do = []
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    
    async with aiohttp.ClientSession() as session:
        for item in data:
            for model_name, response_text in item['responses'].items():
                if f"{item['id']}_{model_name}" not in processed_keys:
                    tasks_to_do.append(evaluate_single_response(session, item, model_name, response_text, system_prompt, semaphore))
        
        print(f"Faltam {len(tasks_to_do)} avaliações. Estimativa de custo: ~${(len(tasks_to_do)*500/1000000)*0.60:.2f}")
        
        with open(OUTPUT_FILE, "a", encoding="utf-8") as f_out:
            for future in tqdm(asyncio.as_completed(tasks_to_do), total=len(tasks_to_do), desc="Processando"):
                result = await future
                if result:
                    f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f_out.flush()
                    os.fsync(f_out.fileno())

if __name__ == "__main__":
    asyncio.run(main())