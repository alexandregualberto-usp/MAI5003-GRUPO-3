import pandas as pd
import numpy as np
import os

TARGET_COLS = [
    'TEstadia', 
    'TAtracado', 
    'TEsperaAtracacao', 
    'TEsperaInicioOp', 
    'TOperacao', 
    'TEsperaDesatracacao'
]

MAIN_TARGET_FOR_ENC = 'TEstadia' 

COL_MERC_EMB = 'Grupo de Mercadoria_embedding_peso'
COL_CONT_EMB = 'Grupo Mercadoria Conteinerizada_embedding_peso'
COLS_ROTA_EMB = ['Origem_embedding_peso', 'Destino_embedding_peso']

COLS_TO_DROP = [
    'IDAtracacao', 
    'ts_atrac', 'ts_atrac_epoch_s', 
    'ano', 'mes_num', 
    'TEstadia_missing', 'TAtracado_missing', 
    'TEsperaAtracao_missing', 'TEsperaInicioOp_missing', 
    'TOperacao_missing', 'TEsperaDesatracacao_missing',
    'Grupo de Mercadoria', 'Grupo Mercadoria Conteinerizada', 
    'Origem', 'Destino', 'Nacionalidade do Armador'
]

COLS_TARGET_ENC = ['Porto Atracação'] 
COLS_OHE = [
    'UF', 'Tipo de Navegação da Atracação', 'Tipo Operação da Carga', 
    'Natureza da Carga', 'Percurso Transporte Interiores', 'Sentido', 
    'Tipo da Autoridade Portuária', 'FlagCabotagem', 'FlagCabotagemMovimentacao', 
    'FlagLongoCurso', 'FlagMCOperacaoCarga', 'FlagTransporteViaInterioir',
    'Instalação Portuária em Rio', 'Carga Geral Acondicionamento', 
    'ConteinerEstado', 'STNaturezaCarga' 
]

def expand_embedding_column(df, col_name):
    if col_name not in df.columns:
        return df, []

    emb_matrix = pd.DataFrame(df[col_name].tolist(), index=df.index)
    
    new_col_names = [f"{col_name}_{i}" for i in range(emb_matrix.shape[1])]
    emb_matrix.columns = new_col_names
    
    df = pd.concat([df, emb_matrix], axis=1)
    df = df.drop(columns=[col_name])
    
    return df, new_col_names

def process_and_save(df_input, filename_prefix="data"):
    
    df = df_input.copy()

    df = df.drop(columns=COLS_TO_DROP, errors='ignore')
    
    for col in COLS_TARGET_ENC:
        if col in df.columns and MAIN_TARGET_FOR_ENC in df.columns:
            
            global_median = df[MAIN_TARGET_FOR_ENC].median()
            
            df[col] = df.groupby(col)[MAIN_TARGET_FOR_ENC].transform('median').fillna(global_median)

    valid_ohe = [c for c in COLS_OHE if c in df.columns]
    df = pd.get_dummies(df, columns=valid_ohe, dummy_na=False)

    df, cols_merc_expanded = expand_embedding_column(df, COL_MERC_EMB)
    df, cols_cont_expanded = expand_embedding_column(df, COL_CONT_EMB)
    
    cols_rota_expanded = []
    for col in COLS_ROTA_EMB:
        df, new_cols = expand_embedding_column(df, col)
        cols_rota_expanded.extend(new_cols)

    df = df.apply(pd.to_numeric, errors='coerce').fillna(0).astype('float32')
    
    available_targets = [t for t in TARGET_COLS if t in df.columns]
    y = df[available_targets].values
    
    X_merc = df[cols_merc_expanded].values if cols_merc_expanded else np.empty((len(df), 0))
    X_cont = df[cols_cont_expanded].values if cols_cont_expanded else np.empty((len(df), 0))
    
    cols_exclude = available_targets + cols_merc_expanded + cols_cont_expanded
    cols_base = [c for c in df.columns if c not in cols_exclude]
    
    X_base = df[cols_base].values

    os.makedirs('train/pre_train/processed_data/', exist_ok=True)
    save_path = f"train/pre_train/processed_data/{filename_prefix}.npz"
    
    np.savez_compressed(
        save_path,
        X_base=X_base,
        X_merc=X_merc,
        X_cont=X_cont,
        y=y,
        cols_base=cols_base,
        target_names=available_targets 
    )
    
    print(f"   [OK] Saved: {save_path}")
    print(f"   Shapes: Base{X_base.shape}, Merc{X_merc.shape}, Cont{X_cont.shape}, Y{y.shape}")
    print(f"   Targets: {available_targets}")
    
    return save_path


if __name__ == "__main__":
    try:
        df_train = pd.read_parquet("train/data/df_train.parquet")
        df_test = pd.read_parquet("train/data/df_test.parquet") 
        df_val = pd.read_parquet("train/data/df_validation.parquet")
        df_train_val = pd.read_parquet("train/data/df_train_validation.parquet")

        process_and_save(df_train, "train_final")
        process_and_save(df_train_val, "train_validation_final")
        process_and_save(df_val, "validation_final")
        process_and_save(df_test, "test_final")
        
        
    except FileNotFoundError as e:
        print(f"\n File not found: {e}")
