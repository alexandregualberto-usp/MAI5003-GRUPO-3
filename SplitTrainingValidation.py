import pandas as pd
from pathlib import Path
import os
import numpy as np


# ============================================================
# CONFIGURAÇÃO
# ============================================================


BASE_DIR = Path("data/aggregated")  # onde estão os anos
OUT_DIR  = Path("data/final_datasets")
OUT_DIR.mkdir(parents=True, exist_ok=True)


FINAL_PATH = OUT_DIR / "df_atracacoes.parquet"


# Split
TRAIN_PATH = OUT_DIR / "df_train_validation.parquet"
TRAIN_FINAL_PATH = OUT_DIR / "df_train.parquet"
VAL_PATH   = OUT_DIR / "df_validation.parquet"
TEST_PATH  = OUT_DIR / "df_test.parquet"




# ============================================================
# 1. Carregar todos os Parquets e adicionar coluna Ano
# ============================================================


all_dfs = []
files = sorted(BASE_DIR.glob("df_*_atracacoes.parquet"))


if not files:
   raise FileNotFoundError(f"Nenhum arquivo encontrado em: {BASE_DIR}")


print(f" Encontrados {len(files)} arquivos (anos).")


for fp in files:
   # extrai o ano do nome do arquivo: df_2019_with_embeddings.parquet
   stem = fp.stem
   ano = int(stem.split("_")[1])


   print(f"→ Lendo {fp.name} (ano={ano})...")


   df = pd.read_parquet(fp)
   df["ano"] = ano
   all_dfs.append(df)


# ============================================================
# 2. Concatena tudo em um único DF
# ============================================================


print(" Concatenando todos os anos...")
df_all = pd.concat(all_dfs, ignore_index=True)


print(f" Dataset final: {df_all.shape[0]:,} linhas, {df_all.shape[1]} colunas")


# ============================================================
# 3. Salva o dataset completo
# ============================================================


print(f" Salvando dataset final em {FINAL_PATH} ...")
df_all.to_parquet(FINAL_PATH, index=False)
print(" Salvo!")




# ============================================================
# 4. Train/Validation/Test split - 80/10/10
# ============================================================


print("  Gerando splits 80/10/10 ...")


# embaralha
df_all = df_all.sample(frac=1.0, random_state=42).reset_index(drop=True)


n = len(df_all)
n_train = int(0.80 * n)
n_val   = int(0.10 * n)
# resto vai para teste
n_test  = n - n_train - n_val


df_train_val = df_all.iloc[:n_train]
df_val   = df_all.iloc[n_train:n_train+n_val]
df_train = df_all.iloc[:n_train+n_val]
df_test  = df_all.iloc[n_train+n_val:]


# ============================================================
# 5. Salva os splits
# ============================================================


df_train_val.to_parquet(TRAIN_PATH, index=False)
df_train.to_parquet(TRAIN_FINAL_PATH, index=False)
df_val.to_parquet(VAL_PATH, index=False)
df_test.to_parquet(TEST_PATH, index=False)


print(" Splits salvos com sucesso!")
print(f"   → Treino (durante a validação)    : {len(df_train_val):,} linhas")
print(f"   → Validação  : {len(df_val):,} linhas")
print(f"   → Treino (pós validação])    : {len(df_train):,} linhas")
print(f"   → Teste      : {len(df_test):,} linhas")


print("\n Processo concluído.")



