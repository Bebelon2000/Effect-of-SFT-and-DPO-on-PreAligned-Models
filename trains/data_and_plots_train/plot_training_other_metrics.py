import json
import matplotlib.pyplot as plt
import os
import math
import pandas as pd  # Necessário para a suavização (rolling mean)

# --- CONFIGURAÇÃO DOS CAMINHOS ---
# Atualize com os seus caminhos reais se necessário
FILES = {
    "SFT": r"C:\Users\berna\scripts_Artigo_Final_1sem\dados_graficos\sft_state.json",
    "SFT_DPO": r"C:\Users\berna\scripts_Artigo_Final_1sem\dados_graficos\dpo_sft_state.json",
    "Just_DPO": r"C:\Users\berna\scripts_Artigo_Final_1sem\dados_graficos\dpo_direct_state.json"
}

OUTPUT_DIR = "graficos_metricas_suavizados"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_log_history(file_path):
    if not os.path.exists(file_path):
        print(f"[ERRO] Ficheiro não encontrado: {file_path}")
        return []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict) and "log_history" in data:
            return data["log_history"]
        elif isinstance(data, list):
            return data
        return []
    except Exception as e:
        print(f"[ERRO] Falha ao ler {file_path}: {e}")
        return []

def save_plot_with_smoothing(steps, values, title, ylabel, color, filename):
    """Gera gráfico com linha 'Raw' transparente e linha 'Smooth' sólida."""
    if not values:
        return

    plt.figure(figsize=(10, 6)) # Tamanho ligeiramente maior para melhor leitura
    
    # 1. Plotar Dados Brutos (Fundo transparente)
    plt.plot(steps, values, color=color, alpha=0.15, linewidth=1, label='Raw Data')
    
    # 2. Calcular Suavização (Média Móvel)
    # Define o tamanho da janela: 5% dos dados ou mínimo de 10 passos
    window_size = max(5, int(len(values) * 0.05))
    
    # Usa Pandas para calcular a média móvel mantendo o alinhamento
    series_smooth = pd.Series(values).rolling(window=window_size, min_periods=1).mean()
    
    # 3. Plotar Dados Suavizados (Linha sólida e mais grossa)
    plt.plot(steps, series_smooth, color=color, alpha=1.0, linewidth=2.5, label='Smoothed')
    
    # Estilização
    plt.title(title, fontsize=15, fontweight='bold')
    plt.xlabel("Training Steps", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='best') # Mostra a legenda (Raw vs Smoothed)
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path, dpi=300)
    print(f"   -> Gráfico salvo: {save_path}")
    plt.close()

def process_sft(name, history):
    print(f"\n--- Processando SFT ({name}) ---")
    if any('mean_token_accuracy' in x for x in history):
        data = [(x['step'], x['mean_token_accuracy']) for x in history if 'mean_token_accuracy' in x]
        if data:
            steps, vals = zip(*data)
            save_plot_with_smoothing(steps, vals, f'{name}: Token Accuracy', 'Accuracy', '#2ca02c', f'{name}_token_accuracy.png')

    if any('entropy' in x for x in history):
        data = [(x['step'], x['entropy']) for x in history if 'entropy' in x]
        if data:
            steps, vals = zip(*data)
            save_plot_with_smoothing(steps, vals, f'{name}: Entropy', 'Entropy', '#9467bd', f'{name}_entropy.png')

    if any('loss' in x for x in history):
        data = []
        for x in history:
            if 'loss' in x:
                try:
                    ppl = math.exp(x['loss'])
                    data.append((x['step'], ppl))
                except OverflowError: continue
        if data:
            steps, vals = zip(*data)
            save_plot_with_smoothing(steps, vals, f'{name}: Perplexity', 'Perplexity', '#d62728', f'{name}_perplexity.png')

def process_dpo(name, history):
    print(f"\n--- Processando DPO ({name}) ---")
    acc_keys = ['rewards/accuracies', 'eval/rewards/accuracies', 'train/rewards/accuracies']
    margin_keys = ['rewards/margins', 'eval/rewards/margins', 'train/rewards/margins']
    
    # Acurácia
    found_acc = next((k for k in acc_keys if any(k in x for x in history)), None)
    if found_acc:
        data = [(x['step'], x[found_acc]) for x in history if found_acc in x]
        if data:
            steps, vals = zip(*data)
            save_plot_with_smoothing(steps, vals, f'{name}: Reward Accuracy', 'Accuracy', '#1f77b4', f'{name}_reward_accuracy.png')

    # Margens
    found_margin = next((k for k in margin_keys if any(k in x for x in history)), None)
    if found_margin:
        data = [(x['step'], x[found_margin]) for x in history if found_margin in x]
        if data:
            steps, vals = zip(*data)
            save_plot_with_smoothing(steps, vals, f'{name}: Reward Margins', 'Margin', '#ff7f0e', f'{name}_reward_margins.png')
            
    # Loss DPO
    if any('loss' in x for x in history):
        data = [(x['step'], x['loss']) for x in history if 'loss' in x]
        if data:
            steps, vals = zip(*data)
            # DPO loss usa uma cor cinza/neutra para diferenciar
            save_plot_with_smoothing(steps, vals, f'{name}: DPO Loss', 'Loss', '#7f7f7f', f'{name}_dpo_loss.png')

if __name__ == "__main__":
    for name, path in FILES.items():
        history = load_log_history(path)
        if not history: continue
        is_dpo = any('rewards/' in k for x in history for k in x.keys())
        if is_dpo:
            process_dpo(name, history)
        else:
            process_sft(name, history)

    print(f"\n\nConcluído! Gráficos suavizados salvos em: {os.path.abspath(OUTPUT_DIR)}")