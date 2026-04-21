import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.ticker as mtick

# --- CONFIGURAÇÃO ---
INPUT_FILE = "judge_evaluations_clean.json"
OUTPUT_DIR = "graficos_micro_42_pro"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# A mesma ordem e cores do script de 4k para consistência visual
MODEL_ORDER = ["Base", "Just_DPO", "SFT", "SFT_DPO"]
COLORS = {
    "hard": "#4c72b0", "none": "#dd8452", "soft": "#55a868", 
    "proper_refusal": "#55a868", "no_refusal": "#4c72b0", "over_refusal": "#dd8452", 
    "not_applicable": "gray", "safe": "#55a868", "unsafe": "#d62728"
}

def load_data(filepath):
    print(f">>> Lendo ficheiro: {filepath}...")
    try:
        df = pd.read_json(filepath)
        # Garante que helpfulness é número
        if 'helpfulness' in df.columns:
            df['helpfulness'] = pd.to_numeric(df['helpfulness'], errors='coerce')
        return df
    except Exception as e:
        print(f"Erro ao ler JSON: {e}")
        return pd.DataFrame()

def add_labels(ax, is_percentage=False):
    """Adiciona valores acima das barras com margem de segurança."""
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            if is_percentage:
                label = f'{height:.1%}'
            else:
                label = f'{height:.2f}'
            
            # Ajuste fino da posição do texto
            ax.annotate(label, 
                        (p.get_x() + p.get_width() / 2., height), 
                        ha='center', va='bottom', # va='bottom' mete o texto EM CIMA da barra
                        xytext=(0, 3), 
                        textcoords='offset points',
                        fontsize=11, fontweight='bold', color='black')

def plot_unsafe_rate(df):
    plt.figure(figsize=(7, 6))
    
    # Calcula taxa de insegurança
    rates = df.groupby('model')['safety'].apply(lambda x: (x == 'unsafe').mean()).reindex(MODEL_ORDER).reset_index()
    
    ax = sns.barplot(data=rates, x='model', y='safety', order=MODEL_ORDER, color="#d62728")
    
    # --- CORREÇÃO VISUAL: ESCALA DINÂMICA ---
    max_val = rates['safety'].max()
    # Define o limite como o valor máximo + 20% de folga, ou no mínimo 0.15 se for muito baixo
    top_limit = max(0.15, max_val * 1.3)
    plt.ylim(0, top_limit)
        
    plt.title("Taxa de Insegurança (Micro - 42 Prompts)", fontsize=14)
    plt.ylabel("Proporção Insegura", fontsize=12)
    plt.xlabel("")
    
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    add_labels(ax, is_percentage=True)
    
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/micro_unsafe_rate.png", dpi=300)
    plt.close()

def plot_stacked_bar(df, column, title, filename, custom_colors):
    plt.figure(figsize=(8, 6))
    
    # Normaliza para 100%
    ct = pd.crosstab(df['model'], df[column], normalize='index')
    ct = ct.reindex(MODEL_ORDER)
    
    available_cols = [c for c in custom_colors.keys() if c in ct.columns]
    colors_list = [custom_colors[c] for c in available_cols]
    
    ax = ct[available_cols].plot(kind='bar', stacked=True, color=colors_list, figsize=(8, 6), width=0.8)
    
    plt.title(title, fontsize=14)
    plt.ylabel("Proporção", fontsize=12)
    plt.xlabel("")
    # Move a legenda para fora para não tapar o gráfico
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    plt.ylim(0, 1.0)
    
    # Rótulos no meio das barras
    for c in ax.containers:
        # Só mostra texto se a barra for maior que 5% para não encavalar
        labels = [f'{v:.1%}' if v > 0.05 else '' for v in c.datavalues]
        ax.bar_label(c, labels=labels, label_type='center', fontsize=10, color='white', weight='bold')

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{filename}", dpi=300)
    plt.close()

def plot_helpfulness(df):
    plt.figure(figsize=(7, 6))
    
    # Helpfulness com Intervalo de Confiança
    ax = sns.barplot(data=df, x='model', y='helpfulness', order=MODEL_ORDER, 
                     color="#4c72b0", errorbar=('ci', 95), capsize=0.1)
    
    plt.title("Utilidade Média (Micro - 42 Prompts)", fontsize=14)
    plt.ylabel("Pontuação (0-2)", fontsize=12)
    plt.xlabel("")
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # --- CORREÇÃO VISUAL: AUMENTAR O TETO ---
    # Como os valores podem chegar a 2.0 e ainda temos o texto em cima,
    # precisamos de espaço. 2.4 é seguro.
    plt.ylim(0, 2.4) # Aumenta o limite superior para 3.0 para acomodar os rótulos
    
    # Adiciona a média numérica acima da barra
    means = df.groupby('model')['helpfulness'].mean().reindex(MODEL_ORDER) 
    for i, val in enumerate(means):
        # Texto posicionado um pouco acima da barra
        ax.text(i, val + 0.30, f'{val:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/micro_helpfulness.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    df = load_data(INPUT_FILE)
    
    if not df.empty:
        plot_unsafe_rate(df)
        plot_helpfulness(df)
        
        cols_refusal = {k:v for k,v in COLORS.items() if k in ["no_refusal", "over_refusal", "proper_refusal"]}
        plot_stacked_bar(df, 'refusal_type', "Tipos de Recusa (Micro)", "micro_refusal_types.png", cols_refusal)
        
        cols_align = {k:v for k,v in COLORS.items() if k in ["hard", "none", "soft"]}
        plot_stacked_bar(df, 'alignment_style', "Estilo de Alinhamento (Micro)", "micro_alignment_style.png", cols_align)
        
        print(f"✅ Gráficos Micro PRO (Corrigidos) gerados em: {os.path.abspath(OUTPUT_DIR)}")
    else:
        print("Erro: DataFrame vazio.")