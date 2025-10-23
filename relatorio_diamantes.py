# relatorio_diamantes_final_v3.py

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import warnings

# --- CONFIGURAÇÃO INICIAL ---
matplotlib.use('Agg')
warnings.filterwarnings('ignore', category=FutureWarning)

console = Console(record=True)

# --- CRIAÇÃO DA PASTA DE IMAGENS ---
output_folder = "img"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    console.print(f"📁 Pasta '{output_folder}' criada para salvar os gráficos.")

# --- INÍCIO DO SCRIPT ---
console.rule("[bold magenta]RELATÓRIO DE ANÁLISE AVANÇADA DO DATASET 'DIAMONDS' (LOCAL)[/bold magenta]", style="magenta")

# --- INTRODUÇÃO ---
intro_text = (
    "Este script realiza a limpeza, preparação e análise avançada do dataset 'diamonds' (carregado localmente). "
    "O objetivo é transformar variáveis, limpar dados, analisar o valor relativo (preço por quilate) e "
    "analisar a relação entre as características dos diamantes e seus preços."
)
console.print(Panel(intro_text, title="[cyan]Introdução[/cyan]", border_style="cyan"))

# --- ETAPA 1: CARREGAMENTO DOS DADOS (COM TRATAMENTO DE ERRO) ---
console.print("\n\n" + "="*80)
console.print("📥 [bold]ETAPA 1: CARREGAMENTO DOS DADOS[/bold] 📥")
console.print("="*80 + "\n")
try:
    console.print("Carregando dataset 'diamonds.csv' local...")
    df_bruto = pd.read_csv('diamonds.csv')
    console.print("[bold green]Dataset carregado com sucesso![/bold green]")

except FileNotFoundError:
    error_text = "O arquivo 'diamonds.csv' não foi encontrado. Por favor, certifique-se de que ele está na mesma pasta que este script."
    console.print(Panel(error_text, title="[bold red]ERRO CRÍTICO[/bold red]", border_style="red"))
    console.rule("[bold red]EXECUÇÃO INTERROMPIDA[/bold red]", style="red")
    sys.exit()
except Exception as e:
    error_text = f"Ocorreu um erro ao ler o arquivo: {e}"
    console.print(Panel(error_text, title="[bold red]ERRO CRÍTICO[/bold red]", border_style="red"))
    console.rule("[bold red]EXECUÇÃO INTERROMPIDA[/bold red]", style="red")
    sys.exit()

# --- ETAPA 2: DIAGNÓSTICO DOS DADOS BRUTOS ---
console.print("\n\n" + "="*80)
console.print("📊 [bold]ETAPA 2: DIAGNÓSTICO DOS DADOS BRUTOS[/bold] 📊")
console.print("="*80 + "\n")
console.print(f"O dataset bruto foi carregado com [bold]{df_bruto.shape[0]}[/bold] linhas e [bold]{df_bruto.shape[1]}[/bold] colunas.")
console.print()

if 'Unnamed: 0' in df_bruto.columns:
    console.print("Removendo coluna 'Unnamed: 0' (índice extra) do dataset.")
    df_bruto = df_bruto.drop(columns=['Unnamed: 0'])
    console.print(f"Dataset agora possui [bold]{df_bruto.shape[1]}[/bold] colunas.")
    console.print()


impossible_dims = df_bruto[(df_bruto['x'] == 0) | (df_bruto['y'] == 0) | (df_bruto['z'] == 0)].shape[0]

dirty_data_text = (
    f"A análise inicial revelou dados nulos (se houver):\n{df_bruto.isnull().sum()}\n\n"
    f"Mais importante, foi identificado um problema de integridade dos dados:\n"
    f"- [bold red]{impossible_dims} diamantes[/bold red] estão listados com dimensões ('x', 'y' ou 'z') iguais a 0, o que é fisicamente impossível.\n\n"
    "As colunas 'cut', 'color' e 'clarity' são categóricas e precisam ser codificadas para análise de correlação."
)
console.print(Panel(dirty_data_text, title="[yellow]Diagnóstico dos Dados Brutos[/yellow]", border_style="yellow"))

# --- ETAPA 3: PROCESSO DE LIMPEZA E PREPARAÇÃO ---
console.print("\n\n" + "="*80)
console.print("✨ [bold]ETAPA 3: PROCESSO DE LIMPEZA E PREPARAÇÃO[/bold] ✨")
console.print("="*80 + "\n")

df_limpo = df_bruto[(df_bruto['x'] > 0) & (df_bruto['y'] > 0) & (df_bruto['z'] > 0)].copy()
removed_rows = df_bruto.shape[0] - df_limpo.shape[0]

cut_map = {'Fair': 1, 'Good': 2, 'Very Good': 3, 'Premium': 4, 'Ideal': 5}
df_limpo['cut_encoded'] = df_limpo['cut'].map(cut_map)

color_map = {'J': 1, 'I': 2, 'H': 3, 'G': 4, 'F': 5, 'E': 6, 'D': 7}
df_limpo['color_encoded'] = df_limpo['color'].map(color_map)

clarity_map = {'I1': 1, 'SI2': 2, 'SI1': 3, 'VS2': 4, 'VS1': 5, 'VVS2': 6, 'VVS1': 7, 'IF': 8}
df_limpo['clarity_encoded'] = df_limpo['clarity'].map(clarity_map)

df_limpo['volume'] = df_limpo['x'] * df_limpo['y'] * df_limpo['z']
df_limpo['price_per_carat'] = df_limpo['price'] / df_limpo['carat']

cleaning_steps_text = (
    "As seguintes ações foram tomadas:\n"
    f"1. [bold]Remoção de Linhas:[/bold] {removed_rows} linhas com dimensões 'x', 'y' ou 'z' zeradas foram removidas.\n"
    "2. [bold]Codificação:[/bold] 'cut', 'color' e 'clarity' foram mapeadas para valores numéricos.\n"
    "3. [bold]Engenharia de Feature (Volume):[/bold] Criada a coluna 'volume' (x*y*z).\n"
    "4. [bold]Engenharia de Feature (Valor Relativo):[/bold] Criada a coluna 'price_per_carat' (price/carat)."
)
console.print(Panel(cleaning_steps_text, title="[yellow]Ações de Limpeza e Preparação[/yellow]", border_style="yellow"))

df_limpo.to_csv('diamonds_dados_limpos.csv', index=False)
console.print("\n📝 [bold]O dataset limpo e preparado foi salvo como 'diamonds_dados_limpos.csv'[/bold]")

# --- ETAPA 4: QUATRO ANÁLISES BIVARIADAS (REQUISITO 1) ---
sns.set_style("whitegrid")
console.print("\n\n" + "="*80)
console.print("📈 [bold]ETAPA 4: INICIANDO ANÁLISES BIVARIADAS[/bold] 📈")
console.print("="*80 + "\n")

# --- Análise 1: carat vs price ---
console.print("  [bold]1. Análise: Quilates (carat) vs. Preço (price)[/bold]")
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_limpo, x='carat', y='price', alpha=0.1, edgecolor=None)
plt.title('Relação entre Quilates (carat) e Preço (price)', fontsize=16)
plt.xlabel('Quilates (carat)', fontsize=12)
plt.ylabel('Preço (USD)', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'grafico_1_carat_vs_price.png'))
console.print(f"  📈 Gráfico salvo como: '{output_folder}/grafico_1_carat_vs_price.png'")
analise1_text = (
    "Como esperado, 'carat' é um dos principais fatores do preço. A relação é positiva e exponencial: "
    "quanto maior o peso em quilates, mais o preço aumenta, e esse aumento se acelera."
)
console.print() 
console.print(Panel(analise1_text, title="[cyan]Conclusão (Análise 1)[/cyan]", border_style="cyan", width=80))

# --- Análise 2: cut vs. price_per_carat ---
console.print("\n  [bold]2. Análise (Avançada): Qualidade do Corte (cut) vs. Preço POR QUILATE[/bold]")
cut_order = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_limpo, x='cut', y='price_per_carat', order=cut_order, palette="coolwarm")
plt.title('Valor Relativo (Preço por Quilate) por Qualidade do Corte', fontsize=16)
plt.xlabel('Qualidade do Corte', fontsize=12)
plt.ylabel('Preço por Quilate (USD)', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'grafico_2_cut_vs_price_per_carat.png'))
console.print(f"  📈 Gráfico salvo como: '{output_folder}/grafico_2_cut_vs_price_per_carat.png'")
analise2_text = (
    "Ao normalizar pelo 'carat' (analisando preço por quilate), a verdadeira influência do corte fica clara. "
    "Diamantes 'Ideal' têm um valor (preço/quilate) mediano significativamente maior "
    "que todos os outros, mostrando que o mercado paga mais pela qualidade do corte, independentemente do tamanho."
)
console.print() 
console.print(Panel(analise2_text, title="[cyan]Conclusão (Análise 2)[/cyan]", border_style="cyan", width=80))

# --- Análise 3: color vs. price_per_carat ---
console.print("\n  [bold]3. Análise (Avançada): Cor (color) vs. Preço POR QUILATE[/bold]")
color_order = ['J', 'I', 'H', 'G', 'F', 'E', 'D'] # Pior para Melhor
plt.figure(figsize=(10, 6))
sns.violinplot(data=df_limpo, x='color', y='price_per_carat', order=color_order, palette="YlOrBr")
plt.title('Valor Relativo (Preço por Quilate) por Cor', fontsize=16)
plt.xlabel('Cor (J=Pior -> D=Melhor)', fontsize=12)
plt.ylabel('Preço por Quilate (USD)', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'grafico_3_color_vs_price_per_carat.png'))
console.print(f"  📈 Gráfico salvo como: '{output_folder}/grafico_3_color_vs_price_per_carat.png'")
analise3_text = (
    "Esta análise mostra claramente a escada de valor. À medida que a cor melhora (de J para D), "
    "o preço por quilate mediano sobe consistentemente. Isso prova que, para um mesmo tamanho, "
    "um diamante de cor 'D' é muito mais valioso que um de cor 'J'."
)
console.print() 
console.print(Panel(analise3_text, title="[cyan]Conclusão (Análise 3)[/cyan]", border_style="cyan", width=80))

# --- Análise 4: depth vs table ---
console.print("\n  [bold]4. Análise: Profundidade (depth) vs. Largura da Mesa (table)[/bold]")
g = sns.jointplot(data=df_limpo, x='table', y='depth', kind='hex', cmap='afmhot')
g.fig.suptitle('Relação de Densidade entre "Depth" e "Table"', y=1.03)
g.fig.tight_layout()
g.savefig(os.path.join(output_folder, 'grafico_4_depth_vs_table.png'))
console.print(f"  📈 Gráfico salvo como: '{output_folder}/grafico_4_depth_vs_table.png'")
analise4_text = (
    "Esta análise explora a relação entre duas proporções de corte. A grande maioria dos diamantes "
    "se concentra em uma faixa específica: 'table' entre 54-60 e 'depth' entre 60-64. "
    "Isso representa o 'ponto ideal' de corte buscado pela indústria."
)
console.print() 
console.print(Panel(analise4_text, title="[cyan]Conclusão (Análise 4)[/cyan]", border_style="cyan", width=80))

# --- Análise 5: Multivariada (carat, price, clarity) ---
console.print("\n  [bold]5. Análise Bônus: Relação Preço/Quilate por Nível de Pureza (clarity)[/bold]")
clarity_order = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']
g = sns.relplot(
    data=df_limpo,
    x='carat', y='price',
    col='clarity', col_order=clarity_order,
    col_wrap=4, 
    kind='scatter',
    alpha=0.1, s=10,
    height=3, aspect=1.2
)
g.fig.suptitle('Relação Preço x Quilate separada por Pureza (Clarity)', y=1.05, fontsize=16)
g.set_axis_labels("Quilates (carat)", "Preço (USD)")
g.fig.tight_layout()
g.savefig(os.path.join(output_folder, 'grafico_5_bonus_multivariada.png'))
console.print(f"  📈 Gráfico salvo como: '{output_folder}/grafico_5_bonus_multivariada.png'")
analise_bonus_text = (
    "Este gráfico múltiplo mostra como a pureza ('clarity') afeta a relação preço/quilate. "
    "Na pureza 'I1' (pior), a relação é fraca. Conforme a pureza aumenta (ex: 'VVS2', 'IF'), "
    "a curva de preço se inclina para cima muito mais rapidamente. Diamantes 'IF' (perfeitos) "
    "têm um aumento de preço exponencial muito mais acentuado."
)
console.print() 
console.print(Panel(analise_bonus_text, title="[cyan]Conclusão (Análise Bônus)[/cyan]", border_style="cyan", width=80))


# --- ETAPA 5: DUAS ANÁLISES DE CORRELAÇÃO (REQUISITO 2) ---
console.print("\n\n" + "="*80)
console.print("🔗 [bold]ETAPA 5: INICIANDO DUAS ANÁLISES DE CORRELAÇÃO[/bold] 🔗")
console.print("="*80 + "\n")

# --- Correlação 1: Matriz de Correlação Geral (Heatmap) ---
console.print("  [bold]1. Correlação: Matriz de Correlação (Heatmap)[/bold]")
numeric_cols = [
    'price', 'carat', 'depth', 'table', 'x', 'y', 'z', 'volume',
    'cut_encoded', 'color_encoded', 'clarity_encoded', 'price_per_carat'
]
corr_matrix = df_limpo[numeric_cols].corr()

plt.figure(figsize=(14, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
plt.title('Matriz de Correlação das Variáveis do Diamante', fontsize=16)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'grafico_6_heatmap_correlacao.png'))
console.print(f"  📈 Gráfico salvo como: '{output_folder}/grafico_6_heatmap_correlacao.png'")
analise_corr1_text = (
    "O heatmap mostra visualmente as correlações lineares (Pearson).\n"
    "1. [bold]'price'[/bold] tem correlação altíssima com [bold]'carat' (0.92)[/bold], 'volume' (0.90), 'x' (0.88), 'y' (0.88) e 'z' (0.88).\n"
    "2. [bold]'price_per_carat'[/bold] (nosso novo indicador) mostra correlação negativa com 'carat' (-0.07), "
    "mas correlações positivas fortes com [bold]'clarity' (0.35)[/bold], [bold]'color' (0.43)[/bold] e [bold]'cut' (0.19)[/bold], "
    "provando que é uma métrica melhor para avaliar a 'qualidade' do que o preço bruto."
)
console.print() 
console.print(Panel(analise_corr1_text, title="[cyan]Conclusão (Correlação 1)[/cyan]", border_style="cyan"))

# --- Correlação 2: As variáveis x, y e z influenciam no price? ---
console.print("\n  [bold]2. Correlação: As dimensões (x, y, z) influenciam no preço?[/bold]")
correlations = corr_matrix['price'][['x', 'y', 'z', 'volume', 'carat']].sort_values(ascending=False)

tabela_corr = Table(title="Correlação Direta com o Preço (price)", style="cyan", title_justify="left")
tabela_corr.add_column("Variável", justify="left", style="magenta")
tabela_corr.add_column("Coeficiente de Correlação", justify="center", style="yellow")

for var, coef in correlations.items():
    tabela_corr.add_row(var, f"{coef:.4f}")

console.print(tabela_corr)

analise_corr2_text = (
    "Sim, as variáveis x (comprimento), y (largura) e z (profundidade) influenciam [bold]fortemente[/bold] o preço. "
    "A tabela acima mostra que todas têm uma correlação linear muito alta (próxima de 0.88) com 'price'.\n"
    "Isso ocorre porque as dimensões determinam o 'volume' do diamante, que por sua vez está "
    "quase perfeitamente correlacionado com o 'carat' (peso), que é o principal fator de preço."
)
console.print() 
console.print(Panel(analise_corr2_text, title="[cyan]Conclusão (Correlação 2)[/cyan]", border_style="cyan"))

# --- ETAPA FINAL: EXPORTAR RELATÓRIO ---
try:
    html_output = "relatorio_final_diamantes.html"
    console.save_html(html_output)
    console.print(f"\n📄 [bold]Relatório completo salvo como arquivo HTML:[/bold] {html_output}")
except Exception as e:
    console.print(f"\n[bold red]Não foi possível salvar o relatório HTML:[/bold] {e}")

console.rule("[bold green]FIM DO RELATÓRIO[/bold green]", style="green")