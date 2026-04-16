# The Effect of SFT and DPO on the Security of Pre-Aligned Models

## 📌 Abstract

Standard alignment for Large Language Models (LLMs) typically involves Supervised Fine-Tuning (SFT) followed by Direct Preference Optimization (DPO). [cite_start]While effective for base models, the impact on models already undergoing instruction-tuning remains under-explored[cite: 5, 6]. 

[cite_start]Our study identifies the **"Safety Regression"** phenomenon: applying naive SFT to a pre-aligned model (Llama-3.1-8B-Instruct), induces catastrophic forgetting of safety guardrails due to **over-compliance**[cite: 7, 10]. [cite_start]We demonstrate that the sequential pipeline (SFT+DPO) fails to recover safety, while a **"Just DPO"** approach directly on the base model optimizes utility while significantly increasing safety[cite: 11, 12].

---

## 🚀 Key Findings (Macro-Evaluation Results)

[cite_start]Based on a large-scale evaluation of **4,000 prompts** using an **LLM-as-a-Judge** (Llama-3.3-70B) framework[cite: 128, 132]:

| Training Strategy | Unsafe Rate (%) 📉 | Avg. Utility (0-2) 📈 | Safety Outcome |
| :--- | :---: | :---: | :--- |
| **Base Model (Llama-3.1)** | 1.4% | 1.61 | Reference Point |
| **SFT (Helpfulness-only)** | **11.8%** | 1.31 | **Critical Regression** |
| **SFT + DPO (Sequential)** | 11.3% | 1.50 | Failed Recovery |
| **Just DPO (Proposed)** | **1.3%** | **1.62** | **Optimal Alignment** |

> **Primary Conclusion:** For modern pre-aligned models, the SFT stage is often redundant and detrimental. [cite_start]The **Just DPO** approach should be considered the new standard for efficient and safe domain adaptation[cite: 12, 409].

---

## 🛠️ Metodologia e Infraestrutura

[cite_start]Este estudo seguiu a metodologia **Design Science Research (DSR)** [cite: 9, 61][cite_start], estruturada em três fases iterativas para garantir o rigor científico e a validade dos artefactos gerados[cite: 66, 69]:

1.  [cite_start]**Fase de Design:** Seleção do dataset **Anthropic HH-RLHF** (161k exemplos) [cite: 71, 72] [cite_start]e definição da arquitetura **QLoRA** com quantização de 4 bits (NF4) para maximizar a eficiência computacional[cite: 66, 92].
2.  [cite_start]**Fase de Desenvolvimento:** Treino de três variantes do modelo **Llama-3.1-8B-Instruct** (SFT, SFT+DPO e Just DPO) num ambiente de Computação de Alto Desempenho (HPC)[cite: 8, 67].
3.  [cite_start]**Fase de Avaliação:** Implementação de um framework **LLM-as-a-Judge** utilizando o **Llama-3.3-70B** para avaliar a segurança e utilidade das respostas[cite: 9, 100].

### Ambiente Computacional
[cite_start]Devido à elevada exigência de VRAM do algoritmo DPO, os treinos foram realizados no cluster **INCD Cirrus**, utilizando nós equipados com GPUs **NVIDIA A100 (80GB VRAM)**[cite: 80, 82]. [cite_start]A orquestração dos processos foi gerida via sistema **SLURM**, garantindo estabilidade e reprodutibilidade[cite: 84].

---

## 📂 Estrutura do Repositório

O repositório está organizado de forma a facilitar a navegação pelos scripts de treino, logs e dados de avaliação:

* **`graficos_metricas_suavizados/`**: Contém as visualizações das curvas de perda (loss) e acurácia de tokens geradas durante as fases de SFT e DPO.
* **`logs_trains_and_merges/`**: Registos detalhados de cada execução no cluster HPC, incluindo métricas de convergência.
* **`macro_eval/`**: Resultados da avaliação em larga escala (N=4.000 prompts) e análises estatísticas globais[cite: 128].
* **`merges/`**: Scripts e ficheiros resultantes da fusão (merge) das matrizes LoRA com o modelo base em precisão FP16[cite: 232].
* [cite_start]**`micro_eval/`**: Dados da inspeção qualitativa profunda realizada com 42 prompts adversariais (stress testing)[cite: 120].
* **`models/`**: Configurações de arquitetura e checkpoints dos modelos treinados.
* **`trains/`**: Scripts principais de treino para as pipelines de SFT e Otimização de Preferência Direta (DPO).
* **`organizacao_das_pastas.txt`**: Documentação técnica detalhada sobre a hierarquia de ficheiros local.
