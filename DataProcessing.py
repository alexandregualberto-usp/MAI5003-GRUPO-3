def imputa_zero(df):
    '''
    Imputa zero para as colunas 'valor_mov_regiao_total','VLPesoCargaConteinerizada_total',
    'valor_mov_hidrovia_total','valor_mov_rio_total','valor_mov_total_hidrografia'
    '''
    cols = ['valor_mov_regiao_total','VLPesoCargaConteinerizada_total','valor_mov_hidrovia_total','valor_mov_rio_total','valor_mov_total_hidrografia']
    for col in cols:
        df[col] = df[col].fillna(0)
    return df

def imputa_nao_se_aplica(df):
    '''
    Imputa 'Não se aplica' para as colunas 'Grupo Mercadoria Conteinerizada','Grupo de Mercadoria'
    '''	
    cols = ['Grupo Mercadoria Conteinerizada','Grupo de Mercadoria']
    for col in cols:
            df[col] = df[col].fillna('Não se aplica')
    return df



def imputa_mediana(df):
    cols = ['Carga Geral Acondicionamento','ConteinerEstado','Percurso Transporte Interiores','lon','lat','ts_atrac','ts_atrac_epoch_s']
    for col in cols:

        df[col] = df.groupby("Porto Atracação")[col].transform(lambda x: x.fillna(x.median()))
        #se por algum motivo não tiver a mediana do porto, coloca a mediana global
        df[col] = df[col].fillna(df[col].median())

    print(f" Coluna '{col}' imputada com a mediana do Porto")
    return df

# Data processing
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import numpy as np
import torch

#tabelas transacionais
bases = ['Carga','Atracacao','TemposAtracacao','Carga_Conteinerizada','Carga_Rio','Carga_Hidrovia','Carga_Regiao']
categorias = ['Instalacao_Origem','Instalacao_Destino',
'Mercadoria','MercadoriaConteinerizada']
anos = range(2018,2025)
dfs=[] #vamos guardar os dataframes aqui (já que criamos vários dinamicamente)


#tabelas de mercadorias
cat_dir = './data/categories/'
df_mercadorias = pd.read_csv(f'{cat_dir}Mercadoria.txt', sep=';')
df_mercadorias_containerizadas = pd.read_csv(f'{cat_dir}/MercadoriaConteinerizada.txt', sep=';')
df_instalcao_destino = pd.read_csv(f'{cat_dir}/Instalacao_Destino.txt', sep=';')


df_instalcao_origem = pd.read_csv(f'{cat_dir}/Instalacao_Origem.txt', sep=';')
[dfs.append(l) for l in ['df_mercadorias','df_mercadorias_containerizadas',
'df_instalcao_destino','df_instalcao_origem']]


#e aqui uma pequena automação para ler ano a ano.
for ano in anos:
   for base in bases:
       nome_df = f'df_{base.lower()}_{ano}'
       dfs.append(nome_df)
       vars()[nome_df] = pd.read_csv(f'./data/{ano}/{ano}{base}.txt', sep=';')


for ano in anos:
   nome_df = f'df_{ano}'
  
   query_df = f"""
{nome_df} = pd.merge(
                   pd.merge(
                       pd.merge(
                           pd.merge(
   pd.merge(
       df_carga_{ano},
       df_atracacao_{ano},
       how='inner',
       on='IDAtracacao',
       suffixes=('_carga', '_atracacao')
   ),
   df_temposatracacao_{ano},
   how='inner',
   on='IDAtracacao'
),
df_mercadorias,
how='inner',
on='CDMercadoria'
),
df_carga_conteinerizada_{ano},
how='left',
on='IDCarga'
),
df_mercadorias_containerizadas,
how='left',
on='CDMercadoriaConteinerizada'
)  
"""
   exec(query_df)
   dfs.append(vars()[nome_df])

# Depois, ao tentar aplicar embeddings, o programa quebrava. Simplesmente estourava memória - não a memória de GPU e sim a própria memória RAM.

embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device='cuda')
def gera_embeddings(df, coluna_texto):
   textos = df[coluna_texto].astype(str).tolist()
   embeddings = embedding_model.encode(textos, show_progress_bar=True)
   df[f'{coluna_texto} embeddings'] = embeddings.tolist()
   del embeddings
   torch.cuda.empty_cache()
   return df
import os, numpy as np, pandas as pd

N_PARTES = 50
PICKLE = './data/processed/df_2018_with_embeddings.pickle'
COL_TXT = 'Grupo de Mercadoria'
COL_EMB = f'{COL_TXT} embeddings'


df_2018 = df_2018.reset_index(drop=True)
subdfs = np.array_split(df_2018, N_PARTES)


# estado atual
if os.path.exists(PICKLE):
   df_proc = pd.read_pickle(PICKLE).reset_index(drop=True)
else:
   df_proc = pd.DataFrame(columns=df_2018.columns.tolist() + [COL_EMB])


linhas_ok = len(df_proc)


# acha a parte a retomar
acum, start = 0, 0
for i, p in enumerate(subdfs):
   if linhas_ok <= acum + len(p) - 1:
       start = i; break
   acum += len(p)


# já terminou?
if start >= N_PARTES:
   print('Nada a fazer. Já está completo.')
else:
   # corta a parte já parcialmente feita (se houver)
   if linhas_ok > acum:
       subdfs[start] = subdfs[start].iloc[(linhas_ok - acum):]


   for i in range(start, N_PARTES):
       print(f'Processando parte {i+1}/{N_PARTES} ({len(subdfs[i])} linhas)')
       part = gera_embeddings(subdfs[i], COL_TXT)
       df_proc = pd.concat([df_proc, part], ignore_index=True)
       os.makedirs(os.path.dirname(PICKLE), exist_ok=True)
       df_proc.to_pickle(PICKLE)


print('Pronto:', PICKLE)





# Processamento com DuckDB


import os, gc
from pathlib import Path
import duckdb
import pandas as pd
import numpy as np


BASE_DIR = Path("./data")
CATEGORIES_DIR = BASE_DIR / "categories"
PROCESSED_DIR = BASE_DIR / "processed"
EMB_DIR = BASE_DIR / "embeddings"


PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
EMB_DIR.mkdir(parents=True, exist_ok=True)


ANOS = list(range(2018, 2025))


import os
import duckdb
from pathlib import Path


def processa_ano_duckdb(ano: int):
   ydir = BASE_DIR / str(ano)
   if not ydir.exists():
       raise FileNotFoundError(f"Pasta do ano {ano} não encontrada: {ydir}")


   out = PROCESSED_DIR / f"df_{ano}.parquet"
   out.parent.mkdir(parents=True, exist_ok=True)


   con = duckdb.connect(database=':memory:')


   # pragmas de desempenho e memória
   con.execute(f"PRAGMA threads={os.cpu_count()}")
   try:
       pages = os.sysconf("SC_PHYS_PAGES")
       page_size = os.sysconf("SC_PAGE_SIZE")
       total_bytes = pages * page_size
       limit_bytes = int(total_bytes * 0.75)
       mem_gib = max(2, limit_bytes // (1024**3))
       con.execute(f"PRAGMA memory_limit='{mem_gib}GiB'")
   except Exception:
       con.execute("PRAGMA memory_limit='8GiB'")
   tmpdir = (PROCESSED_DIR / "_duckdb_tmp")
   tmpdir.mkdir(parents=True, exist_ok=True)
   con.execute(f"PRAGMA temp_directory='{tmpdir.as_posix()}'")


   def read_all_varchar(path):
       return f"SELECT * FROM read_csv_auto('{path}', delim=';', all_varchar=1)"


   # caminhos
   carga_fp  = (ydir / f"{ano}Carga.txt").as_posix()
   atrac_fp  = (ydir / f"{ano}Atracacao.txt").as_posix()
   tempos_fp = (ydir / f"{ano}TemposAtracacao.txt").as_posix()
   cconte_fp = (ydir / f"{ano}Carga_Conteinerizada.txt").as_posix()
   carga_hidrovia_fp = (ydir / f"{ano}Carga_Hidrovia.txt").as_posix()
   carga_rio_fp = (ydir / f"{ano}Carga_Rio.txt").as_posix()
   carga_regiao_fp = (ydir / f"{ano}Carga_Regiao.txt").as_posix()
   merc_fp   = (CATEGORIES_DIR / "Mercadoria.txt").as_posix()
   mercadorias_conteinerizadas_fp = (CATEGORIES_DIR / "MercadoriaConteinerizada.txt").as_posix()
   instalacao_Origem_fp = (CATEGORIES_DIR / "Instalacao_Origem.txt").as_posix()
   instalacao_Destino_fp = (CATEGORIES_DIR / "Instalacao_Destino.txt").as_posix()


   # tabelas base
   con.execute(f"CREATE OR REPLACE TABLE carga AS {read_all_varchar(carga_fp)}")
   con.execute(f"CREATE OR REPLACE TABLE atracacao AS {read_all_varchar(atrac_fp)}")
   con.execute(f"CREATE OR REPLACE TABLE tempos AS {read_all_varchar(tempos_fp)}")
   con.execute(f"CREATE OR REPLACE TABLE carga_conteinerizada AS {read_all_varchar(cconte_fp)}")
   con.execute(f"CREATE OR REPLACE TABLE mercadoria AS {read_all_varchar(merc_fp)}")
   con.execute(f"CREATE OR REPLACE TABLE mercadoria_conteinerizada AS {read_all_varchar(mercadorias_conteinerizadas_fp)}")
   con.execute(f"CREATE OR REPLACE TABLE carga_hidrovia AS {read_all_varchar(carga_hidrovia_fp)}")
   con.execute(f"CREATE OR REPLACE TABLE carga_rio AS {read_all_varchar(carga_rio_fp)}")
   con.execute(f"CREATE OR REPLACE TABLE carga_regiao AS {read_all_varchar(carga_regiao_fp)}")
   con.execute(f"CREATE OR REPLACE TABLE instalacao_Origem AS {read_all_varchar(instalacao_Origem_fp)}")
   con.execute(f"CREATE OR REPLACE TABLE instalacao_Destino AS {read_all_varchar(instalacao_Destino_fp)}")


   # query principal
   query = f"""
   COPY (
     WITH reg_sum AS (
       SELECT IDCarga,
              SUM(TRY_CAST(REPLACE("ValorMovimentado", ',', '.') AS DOUBLE)) AS valor_mov_regiao_total
       FROM carga_regiao GROUP BY IDCarga
     ),
     hid_sum AS (
       SELECT IDCarga,
              SUM(TRY_CAST(REPLACE("ValorMovimentado", ',', '.') AS DOUBLE)) AS valor_mov_hidrovia_total
       FROM carga_hidrovia GROUP BY IDCarga
     ),
     rio_sum AS (
       SELECT IDCarga,
              SUM(TRY_CAST(REPLACE("ValorMovimentado", ',', '.') AS DOUBLE)) AS valor_mov_rio_total
       FROM carga_rio GROUP BY IDCarga
     ),
     tempos_1 AS (
       SELECT IDAtracacao,
              MAX(CASE WHEN "TEsperaAtracacao"='Valor Discrepante' THEN NULL ELSE REPLACE("TEsperaAtracacao",',','.') END) AS "TEsperaAtracacao",
              MAX(CASE WHEN "TEsperaInicioOp"='Valor Discrepante' THEN NULL ELSE REPLACE("TEsperaInicioOp",',','.') END) AS "TEsperaInicioOp",
              MAX(CASE WHEN "TOperacao"='Valor Discrepante' THEN NULL ELSE REPLACE("TOperacao",',','.') END) AS "TOperacao",
              MAX(CASE WHEN "TEsperaDesatracacao"='Valor Discrepante' THEN NULL ELSE REPLACE("TEsperaDesatracacao",',','.') END) AS "TEsperaDesatracacao",
              MAX(CASE WHEN "TAtracado"='Valor Discrepante' THEN NULL ELSE REPLACE("TAtracado",',','.') END) AS "TAtracado",
              MAX(CASE WHEN "TEstadia"='Valor Discrepante' THEN NULL ELSE REPLACE("TEstadia",',','.') END) AS "TEstadia"
       FROM tempos GROUP BY IDAtracacao
     ),
cconte_1 AS (
 SELECT
   IDCarga,
   MIN(CDMercadoriaConteinerizada) AS CDMercadoriaConteinerizada,
   SUM(
     TRY_CAST(
       REPLACE("VLPesoCargaConteinerizada", ',', '.') AS DOUBLE
     )
   ) AS VLPesoCargaConteinerizada_total
 FROM carga_conteinerizada
 GROUP BY IDCarga
),
     mes_map AS (
       SELECT a.IDAtracacao,
              CASE LOWER(TRIM(a."Mes"))
                WHEN 'jan' THEN 1 WHEN 'fev' THEN 2 WHEN 'mar' THEN 3 WHEN 'abr' THEN 4
                WHEN 'mai' THEN 5 WHEN 'jun' THEN 6 WHEN 'jul' THEN 7 WHEN 'ago' THEN 8
                WHEN 'set' THEN 9 WHEN 'out' THEN 10 WHEN 'nov' THEN 11 WHEN 'dez' THEN 12
                ELSE NULL END AS mes_num
       FROM atracacao a
     )
     SELECT
       -- ================= FLAGS =================
       CAST( COALESCE( (TRY_CAST(REPLACE(c."FlagCabotagem", ',', '.') AS DOUBLE) = 1), FALSE) AS BOOLEAN ) AS "FlagCabotagem",
       CAST( COALESCE( (TRY_CAST(REPLACE(c."FlagCabotagemMovimentacao", ',', '.') AS DOUBLE) = 1), FALSE) AS BOOLEAN ) AS "FlagCabotagemMovimentacao",
       CAST( COALESCE( (TRY_CAST(REPLACE(c."FlagLongoCurso", ',', '.') AS DOUBLE) = 1), TRUE) AS BOOLEAN ) AS "FlagLongoCurso",
       c."FlagMCOperacaoCarga",
       c."FlagTransporteViaInterioir",


       -- ================= QUANTIDADES =================
       c."TEU",
       c."QTCarga",


       -- ================= CAMPOS DE LOCALIDADE =================
       CONCAT_WS(', ',
         CASE WHEN TRIM(io."Origem Nome") IN ('', '-', 'Não se aplica','n/a') THEN NULL ELSE TRIM(io."Origem Nome") END,
         CASE WHEN TRIM(io."País Origem") IN ('', '-', 'Não se aplica','n/a') THEN NULL ELSE TRIM(io."País Origem") END,
         CASE WHEN TRIM(io."Continente Origem") IN ('', '-', 'Não se aplica','n/a') THEN NULL ELSE TRIM(io."Continente Origem") END,
         CASE WHEN TRIM(io."BlocoEconomico_Origem") IN ('', '-', 'Não se aplica','n/a') THEN NULL ELSE TRIM(io."BlocoEconomico_Origem") END
       ) AS "Origem",


       CONCAT_WS(', ',
         CASE WHEN TRIM(id."Nome Destino") IN ('', '-', 'Não se aplica','n/a') THEN NULL ELSE TRIM(id."Nome Destino") END,
         CASE WHEN TRIM(id."País Destino") IN ('', '-', 'Não se aplica','n/a') THEN NULL ELSE TRIM(id."País Destino") END,
         CASE WHEN TRIM(id."Continente Destino") IN ('', '-', 'Não se aplica','n/a') THEN NULL ELSE TRIM(id."Continente Destino") END,
         CASE WHEN TRIM(id."BlocoEconomico_Destino") IN ('', '-', 'Não se aplica','n/a') THEN NULL ELSE TRIM(id."BlocoEconomico_Destino") END
       ) AS "Destino",


       -- ================= TEMPOS E VALORES =================
       TRY_CAST(t1."TEstadia" AS DOUBLE) AS "TEstadia",
       TRY_CAST(t1."TAtracado" AS DOUBLE) AS "TAtracado",
       TRY_CAST(t1."TEsperaAtracacao" AS DOUBLE) AS "TEsperaAtracacao",
       TRY_CAST(t1."TEsperaInicioOp" AS DOUBLE) AS "TEsperaInicioOp",
       TRY_CAST(t1."TOperacao" AS DOUBLE) AS "TOperacao",
       TRY_CAST(t1."TEsperaDesatracacao" AS DOUBLE) AS "TEsperaDesatracacao",


       -- ================= CAMPOS DIVERSOS =================
       CAST(ROUND(TRY_CAST(REPLACE(c."VLPesoCargaBruta", ',', '.') AS DOUBLE), 2) AS DECIMAL(18,2)) AS "VLPesoCargaBruta",
       cc.VLPesoCargaConteinerizada_total AS "VLPesoCargaConteinerizada_total",
       c."Tipo Operação da Carga",
       c."Carga Geral Acondicionamento",
       c."ConteinerEstado",
       COALESCE(NULLIF(TRIM(c."Percurso Transporte Interiores"), ''), NULL) AS "Percurso Transporte Interiores",
       c."STNaturezaCarga",
       c."Natureza da Carga",
       c."Sentido",
       TRY_CAST(REPLACE(SPLIT_PART(a."Coordenadas", ',', 1), ',', '.') AS DOUBLE) AS lon,
       TRY_CAST(REPLACE(SPLIT_PART(a."Coordenadas", ',', 2), ',', '.') AS DOUBLE) AS lat,
       TRY_STRPTIME(a."Data Atracação", '%d/%m/%Y %H:%M:%S') AS ts_atrac,
       EPOCH(TRY_STRPTIME(a."Data Atracação", '%d/%m/%Y %H:%M:%S')) AS ts_atrac_epoch_s,
       mm.mes_num,
       SIN(2*PI()*mm.mes_num/12.0) AS mes_sin,
       COS(2*PI()*mm.mes_num/12.0) AS mes_cos,
       COALESCE(a."UF", 'São Paulo') AS "UF",
       CASE WHEN LOWER(TRIM(a."Instalação Portuária em Rio")) IN ('1','s','sim','true','yes','y') THEN 1 ELSE 0 END AS "Instalação Portuária em Rio",
       a."Porto Atracação",
       a."Tipo da Autoridade Portuária",
       a."Ano",
       COALESCE(a."Tipo de Navegação da Atracação", 'Longo Curso') AS "Tipo de Navegação da Atracação",
       m."Grupo de Mercadoria",
       mc."Grupo Mercadoria Conteinerizada",
       CASE
         WHEN TRY_CAST(REPLACE(a."Nacionalidade do Armador", ',', '.') AS DOUBLE)=2 THEN 1
         WHEN TRY_CAST(REPLACE(a."Nacionalidade do Armador", ',', '.') AS DOUBLE)=1 THEN 0
         ELSE 1 END AS "Nacionalidade do Armador",
       reg_sum.valor_mov_regiao_total,
       hid_sum.valor_mov_hidrovia_total,
       rio_sum.valor_mov_rio_total,
       COALESCE(reg_sum.valor_mov_regiao_total,0)+COALESCE(hid_sum.valor_mov_hidrovia_total,0)+COALESCE(rio_sum.valor_mov_rio_total,0) AS valor_mov_total_hidrografia,
       c.IDCarga,
       c.IDAtracacao,
       c.CDMercadoria,
       cc.CDMercadoriaConteinerizada


     FROM carga c
     JOIN atracacao a USING (IDAtracacao)
     LEFT JOIN tempos_1 t1 USING (IDAtracacao)
     LEFT JOIN mercadoria m USING (CDMercadoria)
     LEFT JOIN cconte_1 cc USING (IDCarga)
     LEFT JOIN mercadoria_conteinerizada mc ON cc.CDMercadoriaConteinerizada = mc.CDMercadoriaConteinerizada
     LEFT JOIN reg_sum ON c.IDCarga = reg_sum.IDCarga
     LEFT JOIN hid_sum ON c.IDCarga = hid_sum.IDCarga
     LEFT JOIN rio_sum ON c.IDCarga = rio_sum.IDCarga
     LEFT JOIN mes_map mm ON a.IDAtracacao = mm.IDAtracacao
     LEFT JOIN instalacao_Origem io ON c."Origem" = io."Origem"
     LEFT JOIN instalacao_Destino id ON c."Destino" = id."Destino"
   ) TO '{out.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD);
   """


   con.execute(query)
   con.close()
   print(f"Parquet salvo: {out}")
   return out


# faz para todos os anos 

   try:
       print("Processando", ano)
       processa_ano_duckdb(ano)
   except Exception as e:
       print("Falhou", ano, "->", e)


# Aplicação dos tratamentos
df = imputa_tempos(df)
df = imputa_zero(df)
df = imputa_nao_se_aplica(df)

# Salva os dados limpos
output_path = path.with_name(path.stem + "_cleaned.parquet")
df.to_parquet(output_path, index=False)
print(f"Arquivo salvo em: {output_path}")

# Aplicação versão optimizada dos embeddings

import os, gc
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURAÇÃO
# ============================================================


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
ID_COL = "IDCarga"
BATCH = 128
DEVICE = "cuda"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
TEXT_COLS = [
   "Grupo de Mercadoria",
   "Grupo Mercadoria Conteinerizada",
   "Origem",
   "Destino"
]


# ============================================================
# FUNÇÃO ULTRA-OTIMIZADA
# ============================================================


def embeda_parquet(parquet_in: Path, text_cols: list, emb_dir: Path):
   """
   Lê parquet em streaming (batches), gera embeddings coluna por coluna e
   salva em PARQUET compactado, um arquivo por coluna.
   """


   print(f"📦 Arquivo de entrada: {parquet_in}")


   # ------------------------------------------------------------
   # carrega o modelo uma única vez
   # ------------------------------------------------------------
   print(f"🧠 Carregando modelo: {MODEL_NAME}")
   model = SentenceTransformer(MODEL_NAME, device=DEVICE)
   model.max_seq_length = 256


   # ------------------------------------------------------------
   # ParquetFile para leitura em streaming
   # ------------------------------------------------------------
   pf = pq.ParquetFile(str(parquet_in))


   # cada coluna vira um arquivo parquet único
   base_name = parquet_in.stem
   out_root = emb_dir / f"dataset={base_name}"
   out_root.mkdir(parents=True, exist_ok=True)


   # ============================================================
   # PROCESSA CADA COLUNA
   # ============================================================


   for text_col in text_cols:
       emb_col = f"{text_col}_embedding"
       out_path = out_root / f"{emb_col}.parquet"


       if out_path.exists():
           print(f"  {emb_col} já existe, pulando.")
           continue


       print(f"\n Processando coluna: {text_col}")


       writer = None
       total_rows = 0


       # ------------------------------------------------------------
       # leitura por batches reais
       # ------------------------------------------------------------
       for batch in pf.iter_batches(columns=[ID_COL, text_col], batch_size=50000):


           b = batch.to_pandas()
           ids = b[ID_COL].to_numpy()
           textos = b[text_col].astype(str).fillna("").tolist()


           # --------------------------------------------------------
           # embedding em batches para não estourar GPU
           # --------------------------------------------------------
           for i in range(0, len(textos), BATCH):
               batch_ids = ids[i:i+BATCH]
               batch_txt = textos[i:i+BATCH]


               emb = model.encode(
                   batch_txt,
                   batch_size=BATCH,
                   show_progress_bar=False,
                   convert_to_numpy=True,
                   normalize_embeddings=False,
               ).astype("float32")


               # Arrow-friendly native vector type
               arr = pa.FixedSizeListArray.from_arrays(
                   pa.array(emb.ravel()), emb.shape[1]
               )


               tbl = pa.table({
                   ID_COL: pa.array(batch_ids),
                   emb_col: arr
               })


               # ----------------------------------------------------
               # escrita contínua usando ParquetWriter
               # ----------------------------------------------------
               if writer is None:
                   writer = pq.ParquetWriter(
                       out_path,
                       tbl.schema,
                       compression="zstd"
                   )
               writer.write_table(tbl)
               total_rows += len(batch_ids)


               del emb, tbl, arr
               gc.collect()


       if writer is not None:
           writer.close()


       print(f"{emb_col} concluído ({total_rows} linhas gravadas).")


   print("\nTodas as colunas processadas com sucesso!\n")


for ano in ANOS:
   parquet_in = PROCESSED_DIR / f"df_{ano}.parquet"
   if parquet_in.exists():
       print(f"Embeddings -> {ano}")
       embeda_parquet(parquet_in, TEXT_COLS, EMB_DIR)
   else:
       print(f"Sem parquet para {ano} — pulando")



#Para que o processamento fosse eficiente e estável, não carregamos todas as colunas na hora de gerar embeddings, 
# economizando memória. É preciso agora reconstruir o dataframe original com as embeddings.

import duckdb
from pathlib import Path
import os, gc


BASE_DIR = Path("data")
PROCESSED_DIR = BASE_DIR / "processed"
EMB_DIR = BASE_DIR / "embeddings"
OUTPUT_DIR = BASE_DIR / "processed_with_embeddings"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


ID_COL = "IDCarga"


TEXT_COLS = [
   "Grupo de Mercadoria",
   "Grupo Mercadoria Conteinerizada",
   "Origem",
   "Destino",
]
ANOS = [ano]




def reconstruct_with_duckdb(ano: int):
   parquet_in = PROCESSED_DIR / f"df_{ano}_cleaned.parquet"
   emb_path = EMB_DIR / f"dataset=df_{ano}"
   parquet_out = OUTPUT_DIR / f"df_{ano}_with_embeddings.parquet"


   if not parquet_in.exists():
       print(f" Pulando {ano}: base não existe -> {parquet_in}")
       return
   if not emb_path.exists():
       print(f" Pulando {ano}: embeddings não existem -> {emb_path}")
       return


   print(f"🔄 Reconstituindo {ano} via DuckDB...")


   con = duckdb.connect(database=":memory:")




   # ---------------- PRAGMAS ----------------
   con.execute(f"PRAGMA threads={min(4, os.cpu_count())}")
   con.execute(f"PRAGMA memory_limit='8GB'")
   con.execute("PRAGMA preserve_insertion_order=FALSE")
   con.execute("PRAGMA enable_object_cache=TRUE")
   con.execute("PRAGMA enable_progress_bar=FALSE")




   tmpdir = OUTPUT_DIR / "_duckdb_tmp_join"
   tmpdir.mkdir(parents=True, exist_ok=True)
   con.execute(f"PRAGMA temp_directory='{tmpdir.as_posix()}'")


   # ---------------- BASE ----------------
   con.execute(f"""
       CREATE TABLE base AS
       SELECT * FROM read_parquet('{parquet_in.as_posix()}');
   """)


   # ---------------- LOOP POR COLUNA ----------------
   for text_col in TEXT_COLS:
       emb_col = f"{text_col}_embedding"
       print(f"  • Processando embeddings para coluna: {text_col} -> {emb_col}")


       files = sorted(emb_path.rglob("*.parquet"))
       if not files:
           print(f" Nenhum parquet de embedding encontrado em {emb_path}")
           continue


       # 🔍 Detectar quais arquivos contêm ESTA coluna
       good_files = []
       for f in files:
           cols = duckdb.query(
               f"SELECT * FROM read_parquet('{f.as_posix()}') LIMIT 1"
           ).columns
           if emb_col in cols:
               good_files.append(f)


       if not good_files:
           print(f" Nenhum arquivo contém a coluna {emb_col}. Pulando…")
           continue


       files_list = ", ".join(f"'{f.as_posix()}'" for f in good_files)


       # ---------------- tmp_emb ----------------
       con.execute("DROP TABLE IF EXISTS tmp_emb;")
       con.execute(f"""
           CREATE TABLE tmp_emb AS
           SELECT
               {ID_COL},
               "{emb_col}"
           FROM read_parquet([{files_list}])
           WHERE "{emb_col}" IS NOT NULL;
       """)


       # ---------------- EMB_CLEAN ----------------
       con.execute("DROP TABLE IF EXISTS EMB_CLEAN;")
       con.execute(f"""
           CREATE TABLE EMB_CLEAN AS
           SELECT {ID_COL}, "{emb_col}"
           FROM (
               SELECT
                   {ID_COL},
                   "{emb_col}",
                   ROW_NUMBER() OVER (PARTITION BY {ID_COL} ORDER BY 1) AS rn
               FROM tmp_emb
           )
           WHERE rn = 1;
       """)


       # ---------------- JOIN COM BASE ----------------
       con.execute("DROP TABLE IF EXISTS base2;")
       con.execute(f"""
           CREATE TABLE base2 AS
           SELECT b.*, e."{emb_col}"
           FROM base b
           LEFT JOIN EMB_CLEAN e
             ON b.{ID_COL} = e.{ID_COL};
       """)


       con.execute("DROP TABLE base;")
       con.execute("ALTER TABLE base2 RENAME TO base;")


       con.execute("DROP TABLE tmp_emb;")
       con.execute("DROP TABLE EMB_CLEAN;")


       gc.collect()


   # ---------------- SALVAR ----------------
   print("  Salvando parquet final com embeddings...")
   con.execute(f"""
       COPY base TO '{parquet_out.as_posix()}'
       (FORMAT PARQUET, COMPRESSION ZSTD);
   """)


   con.close()
   print(f" {ano} salvo em {parquet_out}")




# ---------------- RODAR ----------------
for ano in ANOS:
   reconstruct_with_duckdb(ano)


print(" Reconstrução concluída!")


