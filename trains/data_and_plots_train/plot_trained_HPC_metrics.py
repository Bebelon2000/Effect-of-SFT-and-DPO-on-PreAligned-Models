import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

# --- CONFIGURAÇÃO ---
DATA_DIR = "dados_graficos"
FILES = {
    "SFT": os.path.join(DATA_DIR, "sft_state.json"),
    "Just_DPO": os.path.join(DATA_DIR, "dpo_direct_state.json"),
    "SFT_DPO": os.path.join(DATA_DIR, "dpo_sft_state.json")
}

# Cores Acadêmicas
COLORS = {
    "SFT": "#1f77b4",      # Azul Sóbrio
    "Just_DPO": "#2ca02c", # Verde (Sucesso/Ours)
    "SFT_DPO": "#ff7f0e"   # Laranja (Comparação)
}

def load_data(file_path):
    if not os.path.exists(file_path):
        print(f"ERRO: {file_path} não encontrado.")
        return None
    with open(file_path, 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data['log_history'])

def plot_single_loss(df, model_name, title, filename):
    """Gera um gráfico limpo apenas de Loss para um modelo"""
    if 'loss' not in df.columns: return

    plt.figure(figsize=(8, 5))
    sns.set_style("whitegrid")
    
    data = df[df['loss'].notna()]
    
    # Plot dados originais (bem transparente)
    plt.plot(data['step'], data['loss'], alpha=0.15, color=COLORS[model_name])
    
    # Plot média móvel (Suavização forte para ficar bonito)
    # Window maior = linha mais lisa
    window = 20 if len(data) > 1000 else 5
    plt.plot(data['step'], data['loss'].rolling(window=window).mean(), 
             color=COLORS[model_name], linewidth=2.5, label="Training Loss")

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Training Steps", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    save_path = os.path.join(DATA_DIR, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Salvo: {filename}")

def plot_accuracy_comparison(dfs):
    """Gera o gráfico comparativo de Acurácia (O mais importante)"""
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # Procura colunas de acurácia
    acc_cols = ['rewards/accuracies', 'eval_rewards/accuracies', 'train_accuracy']
    
    for name, df in dfs.items():
        if name == "SFT": continue # SFT não tem acurácia de preferência
        
        col_name = next((c for c in acc_cols if c in df.columns), None)
        if col_name:
            data = df[df[col_name].notna()]
            # Suavização para comparação clara
            plt.plot(data['step'], data[col_name].rolling(window=25).mean(), 
                     color=COLORS[name], linewidth=2.5, label=f"{name} (Smoothed)")
            # Linha original transparente
            plt.plot(data['step'], data[col_name], alpha=0.15, color=COLORS[name])

    plt.title("Evolução da Acurácia de Preferência (DPO)", fontsize=14, fontweight='bold')
    plt.xlabel("Training Steps", fontsize=12)
    plt.ylabel("Acurácia (Prob. Chosen > Rejected)", fontsize=12)
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    save_path = os.path.join(DATA_DIR, "comparacao_acuracia_dpo.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Salvo: comparacao_acuracia_dpo.png")

if __name__ == "__main__":
    dfs = {}
    for name, path in FILES.items():
        dfs[name] = load_data(path)

    # 1. Gerar Gráficos Individuais de Loss
    if dfs["SFT"] is not None:
        plot_single_loss(dfs["SFT"], "SFT", "Convergência de Treino: SFT", "fig_loss_sft.png")
    
    if dfs["Just_DPO"] is not None:
        plot_single_loss(dfs["Just_DPO"], "Just_DPO", "Convergência de Treino: Just DPO", "fig_loss_just_dpo.png")

    if dfs["SFT_DPO"] is not None:
        plot_single_loss(dfs["SFT_DPO"], "SFT_DPO", "Convergência de Treino: SFT + DPO", "fig_loss_sft_dpo.png")

    # 2. Gerar Comparativo de Acurácia
    plot_accuracy_comparison(dfs)