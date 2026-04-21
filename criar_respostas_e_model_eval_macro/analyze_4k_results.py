import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import matplotlib.ticker as mtick

# --- CONFIGURAÇÃO ---
INPUT_FILE = "evaluation_results_4k.jsonl"
OUTPUT_DIR = "graficos_macro_4k_pro"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Ordem lógica
MODEL_ORDER = ["Base", "Just_DPO", "SFT", "SFT_DPO"]
# Cores
COLORS = {"hard": "#4c72b0", "none": "#dd8452", "soft": "#55a868", 
          "proper_refusal": "#55a868", "no_refusal": "#4c72b0", "over_refusal": "#dd8452", 
          "not_applicable": "gray", "safe": "#55a868", "unsafe": "#d62728"}

def load_data(filepath):
    data = []
    print(f">>> Lendo ficheiro: {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                record = json.loads(line)
                eval_data = record.get('evaluation', {})
                if isinstance(eval_data, str): eval_data = json.loads(eval_data)
                
                row = {
                    'model': record.get('model'),
                    'safety': eval_data.get('safety'),
                    'refusal_type': eval_data.get('refusal_type'),
                    'helpfulness': pd.to_numeric(eval_data.get('helpfulness'), errors='coerce'),
                    'alignment_style': eval_data.get('alignment_style')
                }
                data.append(row)
            except: continue
    return pd.DataFrame(data)

def add_labels(ax, is_percentage=False):
    """Adiciona os valores exatos em cima das barras."""
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            if is_percentage:
                label = f'{height:.1%}' # Ex: 14.2%
            else:
                label = f'{height:.2f}' # Ex: 1.15
            
            ax.annotate(label, 
                        (p.get_x() + p.get_width() / 2., height), 
                        ha='center', va='center', 
                        xytext=(0, 5), 
                        textcoords='offset points',
                        fontsize=10, fontweight='bold', color='black')

def plot_unsafe_rate(df):
    plt.figure(figsize=(7, 6))
    
    # Prepara dados
    rates = df.groupby('model')['safety'].apply(lambda x: (x == 'unsafe').mean()).reset_index()
    
    # Gráfico
    ax = sns.barplot(data=rates, x='model', y='safety', order=MODEL_ORDER, color="#d62728")
    
    # MELHORIA: Ajuste de escala dinâmico (não força 1.0 se for tudo baixo)
    # Adiciona 10% de margem no topo para caber o número
    max_val = rates['safety'].max()
    plt.ylim(0, max_val * 1.25) 
    
    plt.title("Taxa de Insegurança (Unsafe Rate)", fontsize=14)
    plt.ylabel("Proporção Insegura", fontsize=12)
    plt.xlabel("")
    
    # Formata eixo Y para percentagem
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    
    # Adiciona os números nas barras
    add_labels(ax, is_percentage=True)
    
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/1_unsafe_rate_pro.png", dpi=300)
    plt.close()

def plot_stacked_bar(df, column, title, filename, custom_colors):
    """Função genérica para gráficos empilhados com rótulos."""
    plt.figure(figsize=(8, 6))
    
    ct = pd.crosstab(df['model'], df[column], normalize='index')
    ct = ct.reindex(MODEL_ORDER)
    
    available_cols = [c for c in custom_colors.keys() if c in ct.columns]
    colors_list = [custom_colors[c] for c in available_cols]
    
    ax = ct[available_cols].plot(kind='bar', stacked=True, color=colors_list, figsize=(8, 6), width=0.8)
    
    plt.title(title, fontsize=14)
    plt.ylabel("Proporção", fontsize=12)
    plt.xlabel("")
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.ylim(0, 1.0)
    
    # Adiciona rótulos no meio das barras (apenas se a barra for grande o suficiente)
    for c in ax.containers:
        # Formata labels: exibe apenas se > 5%
        labels = [f'{v:.1%}' if v > 0.05 else '' for v in c.datavalues]
        ax.bar_label(c, labels=labels, label_type='center', fontsize=9, color='white', weight='bold')

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{filename}", dpi=300)
    plt.close()

def plot_helpfulness_with_error(df, safe_only=False):
    plt.figure(figsize=(7, 6))
    
    if safe_only:
        data_source = df[df['safety'] == 'safe']
        title = "Utilidade (Apenas Respostas Seguras)"
        filename = "4_safe_helpfulness_pro.png"
    else:
        data_source = df
        title = "Utilidade Média Global"
        filename = "3_avg_helpfulness_pro.png"

    # Barplot com Intervalo de Confiança (errorbar='ci') - Padrão do Seaborn
    ax = sns.barplot(data=data_source, x='model', y='helpfulness', order=MODEL_ORDER, 
                     color="#4c72b0", errorbar=('ci', 95), capsize=0.1)
    
    plt.title(title, fontsize=14)
    plt.ylabel("Pontuação (0-2)", fontsize=12)
    plt.ylim(0, 2) # Mantém escala Likert
    plt.xlabel("")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Adiciona labels (um pouco complexo com error bars, vamos colocar a média)
    # Calculamos as médias manualmente para posicionar o texto
    means = data_source.groupby('model')['helpfulness'].mean().reindex(MODEL_ORDER)
    for i, val in enumerate(means):
        ax.text(i, val + 0.05, f'{val:.2f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{filename}", dpi=300)
    plt.close()

if __name__ == "__main__":
    df = load_data(INPUT_FILE)
    if not df.empty:
        # 1. Unsafe Rate (Escala ajustada)
        plot_unsafe_rate(df)
        
        # 2. Alignment Style (Stacked 100%)
        cols_align = {k:v for k,v in COLORS.items() if k in ["hard", "none", "soft"]}
        plot_stacked_bar(df, 'alignment_style', "Estilo de Alinhamento", "2_alignment_style_pro.png", cols_align)
        
        # 3. Helpfulness (Com barras de erro)
        plot_helpfulness_with_error(df, safe_only=False)
        plot_helpfulness_with_error(df, safe_only=True)
        
        # 4. Refusal Type (Stacked 100%)
        cols_refusal = {k:v for k,v in COLORS.items() if k in ["no_refusal", "over_refusal", "proper_refusal"]}
        plot_stacked_bar(df, 'refusal_type', "Tipos de Recusa", "5_refusal_type_pro.png", cols_refusal)
        
        print(f"\n✅ Gráficos PRO gerados em: {OUTPUT_DIR}")
    else:
        print("Erro: Sem dados.")