"""
Análise Exploratória de Dados (EDA) - Ecommerce Estatística
Autor: Felipe Aiasse Franco
Descrição:
    Este script realiza uma análise exploratória completa do dataset
    'ecommerce_estatistica.csv', incluindo visualizações como:
    - Histograma
    - Dispersão
    - Mapa de Calor
    - Gráfico de Barras
    - Gráfico de Pizza
    - Densidade
    - Regressão Linear
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Configurações gerais
sns.set(style="whitegrid", palette="muted")
plt.rcParams["figure.figsize"] = (10, 8)

# -----------------------------------------------------------
# 1. Leitura e preparação dos dados
# -----------------------------------------------------------

df = pd.read_csv("ecommerce_estatistica.csv")
print(df.head())

# Remover coluna de índice exportado
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

print(df.info())
print(df.describe())

# -----------------------------------------------------------
# 2. Histograma - Distribuição de Preços
# -----------------------------------------------------------

plt.figure(figsize=(10, 8))
plt.hist(df["Preço"], bins=100, color="green")
plt.title("Distribuição de Preços dos Produtos")
plt.xlabel("Preço (R$)")
plt.ylabel("Quantidade de Produtos")
plt.tight_layout()
plt.show()

# -----------------------------------------------------------
# 3. Dispersão - Preço vs Nota
# -----------------------------------------------------------

plt.figure(figsize=(10, 8))
sns.scatterplot(data=df, x="Preço", y="Nota")
plt.title("Relação entre Preço e Nota dos Produtos")
plt.xlabel("Preço (R$)")
plt.ylabel("Nota")
plt.tight_layout()
plt.show()

# -----------------------------------------------------------
# 4. Mapa de Calor - Correlações
# -----------------------------------------------------------

cols_corr = [
    "Nota", "N_Avaliações", "Desconto", "Preço",
    "Nota_MinMax", "N_Avaliações_MinMax", "Desconto_MinMax",
    "Preço_MinMax", "Marca_Cod", "Material_Cod",
    "Temporada_Cod", "Qtd_Vendidos_Cod"
]

corr = df[cols_corr].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Mapa de Calor das Correlações")
plt.tight_layout()
plt.show()

# -----------------------------------------------------------
# 5. Gráfico de Barras - Marcas com maior volume de vendas
# -----------------------------------------------------------

vendas_marca = (
    df.groupby("Marca")["Qtd_Vendidos_Cod"]
      .sum()
      .sort_values(ascending=False)
      .head(10)
)

plt.figure(figsize=(10, 8))
sns.barplot(x=vendas_marca.index, y=vendas_marca.values)
plt.title("Top 10 Marcas por Volume de Vendas (Qtd_Vendidos_Cod)")
plt.xlabel("Marca")
plt.ylabel("Volume de Vendas")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# -----------------------------------------------------------
# 6. Gráfico de Pizza - Proporção por Gênero
# -----------------------------------------------------------

counts = df["Gênero"].value_counts()

# Agrupar categorias pequenas (<10%)
limite = counts.sum() * 0.10
counts_corrigido = counts.copy()
counts_corrigido[counts < limite] = 0

outros = counts[counts < limite].sum()
counts_corrigido = counts_corrigido[counts_corrigido > 0]
counts_corrigido["Sem Gênero/Infantil"] = outros

plt.figure(figsize=(10, 8))
plt.pie(
    counts_corrigido,
    labels=counts_corrigido.index,
    autopct="%1.1f%%",
    startangle=90,
    pctdistance=0.8,
    textprops={"fontsize": 12}
)
plt.title("Proporção de Produtos por Gênero")
plt.tight_layout()
plt.show()

# -----------------------------------------------------------
# 7. Densidade - Preço
# -----------------------------------------------------------

plt.figure(figsize=(10, 8))
sns.kdeplot(data=df, x="Preço", fill=True)
plt.title("Densidade de Preços dos Produtos")
plt.xlabel("Preço (R$)")
plt.ylabel("Densidade")
plt.tight_layout()
plt.show()

# -----------------------------------------------------------
# 8. Regressão - Preço vs Nota
# -----------------------------------------------------------

plt.figure(figsize=(10, 8))
sns.regplot(data=df, x="Preço", y="Nota", scatter_kws={"alpha": 0.5})
plt.title("Relação entre Preço e Nota com Linha de Regressão")
plt.xlabel("Preço (R$)")
plt.ylabel("Nota Média")
plt.tight_layout()
plt.show()


