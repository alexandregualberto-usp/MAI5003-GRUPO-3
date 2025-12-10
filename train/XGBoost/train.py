import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import os
import random
import gc
import json
import sys

TARGETS_OF_INTEREST = []  # Empty = Run all targets
N_ESTIMATORS = 3000   
EARLY_STOPPING_ROUNDS = 50
SEEDS = range(1, 11) 
MODELS_DIR = "train/XGBoost/models_json" 

os.makedirs(MODELS_DIR, exist_ok=True)

try:
    xgb.XGBRegressor(device="cuda", n_estimators=1).fit(np.array([[1]]), np.array([0]))
    DEVICE = "cuda"
except Exception as e:
    DEVICE = "cpu"

try:
    data_tr = np.load("train/pre_train/processed_data/train_final.npz", allow_pickle=True)
    data_ts = np.load("train/pre_train/processed_data/test_final.npz", allow_pickle=True)
    data_vl = np.load("train/pre_train/processed_data/validation_final.npz", allow_pickle=True)
except FileNotFoundError:
    print("ERROR: Files not found in 'train/pre_train/processed_data/'.")
    sys.exit()

X_base_tr = data_tr['X_base'].astype(np.float32)
X_merc_tr = data_tr['X_merc'].astype(np.float32)
X_cont_tr = data_tr['X_cont'].astype(np.float32)
y_tr_all  = data_tr['y'].astype(np.float32)
cols_tr   = data_tr['cols_base']

X_base_ts = data_ts['X_base'].astype(np.float32)
X_merc_ts = data_ts['X_merc'].astype(np.float32)
X_cont_ts = data_ts['X_cont'].astype(np.float32)
y_ts_all  = data_ts['y'].astype(np.float32)
cols_ts   = data_ts['cols_base']

X_base_vl = data_vl['X_base'].astype(np.float32)
X_merc_vl = data_vl['X_merc'].astype(np.float32)
X_cont_vl = data_vl['X_cont'].astype(np.float32)
y_vl_all  = data_vl['y'].astype(np.float32)
cols_vl   = data_vl['cols_base']

target_names = data_tr['target_names']

del data_tr, data_ts, data_vl
gc.collect()


def align_columns(X_target, cols_target, target_rows):
    if not np.array_equal(cols_tr, cols_target):
        missing = list(set(cols_tr) - set(cols_target))
        if missing:
            filler = np.zeros((target_rows, len(missing)), dtype=np.float32)
            X_target = np.hstack([X_target, filler])
    return X_target

X_base_ts = align_columns(X_base_ts, cols_ts, X_base_ts.shape[0])
X_base_vl = align_columns(X_base_vl, cols_vl, X_base_vl.shape[0])

scaler = StandardScaler()
X_base_tr = scaler.fit_transform(X_base_tr).astype(np.float32)
X_base_ts = scaler.transform(X_base_ts).astype(np.float32)
X_base_vl = scaler.transform(X_base_vl).astype(np.float32)


results = []
targets_to_run = [t for t in TARGETS_OF_INTEREST if t in target_names]
if not targets_to_run:
    targets_to_run = target_names

for target_name in targets_to_run:
    print(f"\n{'='*60}")
    print(f"TARGET: {target_name}")
    print(f"{'='*60}")

    best_rmse_for_target = float('inf')
    
    gc.collect()

    json_path = f"train/XGBoost/hyperparams/best_params_{target_name}.json"
    current_params = {}
    
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            current_params = json.load(f)
        print(f"   -> Loaded params from {json_path}")
    else:
        print("   -> Using default params (File not found)")
        current_params = {'learning_rate': 0.1, 'max_depth': 6}

    current_params.update({
        'objective': 'reg:squarederror',
        'device': DEVICE,
        'tree_method': 'hist', 
        'n_jobs': -1,
    })
    
    if DEVICE == "cuda" and 'max_bin' not in current_params:
        current_params['max_bin'] = 256 

    current_params.pop('n_estimators', None)

    try:
        idx = np.where(target_names == target_name)[0][0]
        y_tr_curr = y_tr_all[:, idx]
        y_ts_curr = y_ts_all[:, idx]
        y_vl_curr = y_vl_all[:, idx]
    except IndexError:
        print(f"Target index error. Skipping.")
        continue

    for seed in SEEDS:
        print(f"   -> Seed {seed:02d} | Scenarios: ", end="")
        
        random.seed(seed)
        np.random.seed(seed)
        
        scenarios = ['C1', 'C2', 'C3', 'C4']
        
        for sc in scenarios:
            print(f"{sc}..", end=" ")
            
            if sc == 'C1':   
                X_tr, X_ts, X_vl = X_base_tr, X_base_ts, X_base_vl
            elif sc == 'C2': 
                X_tr = np.hstack([X_base_tr, X_merc_tr])
                X_ts = np.hstack([X_base_ts, X_merc_ts])
                X_vl = np.hstack([X_base_vl, X_merc_vl])
            elif sc == 'C3': 
                X_tr = np.hstack([X_base_tr, X_cont_tr])
                X_ts = np.hstack([X_base_ts, X_cont_ts])
                X_vl = np.hstack([X_base_vl, X_cont_vl])
            elif sc == 'C4': 
                X_tr = np.hstack([X_base_tr, X_merc_tr, X_cont_tr])
                X_ts = np.hstack([X_base_ts, X_merc_ts, X_cont_ts])
                X_vl = np.hstack([X_base_vl, X_merc_vl, X_cont_vl])
            
            model = xgb.XGBRegressor(
                n_estimators=N_ESTIMATORS,
                early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                random_state=seed,
                **current_params
            )
            
            model.fit(
                X_tr, y_tr_curr,
                eval_set=[(X_vl, y_vl_curr)], 
                verbose=False
            )
            
            y_pred = model.predict(X_ts)
            current_rmse = np.sqrt(mean_squared_error(y_ts_curr, y_pred))
            
            results.append({
                'Target': target_name,
                'Seed': seed, 
                'Cenario': sc, 
                'R2': r2_score(y_ts_curr, y_pred), 
                'RMSE': current_rmse,
                'MAE': mean_absolute_error(y_ts_curr, y_pred)
            })
            
            if current_rmse < best_rmse_for_target:
                best_rmse_for_target = current_rmse
                
                model_filename = f"{MODELS_DIR}/best_{target_name}.json"
                model.save_model(model_filename)
                
                with open(f"{MODELS_DIR}/best_{target_name}_info.txt", "w") as f:
                    f.write(f"Scenario: {sc}\n")
                    f.write(f"Seed: {seed}\n")
                    f.write(f"RMSE: {current_rmse:.6f}\n")
                    f.write(f"N_Features: {X_tr.shape[1]}\n")
            
            del model, X_tr, X_ts, X_vl
            gc.collect() 
            
        print("OK")

df_res = pd.DataFrame(results)

if not df_res.empty:
    print("\n" + "="*60)
    print("FINAL SUMMARY (Average per Scenario)")
    print("="*60)
    summary = df_res.groupby(['Target', 'Cenario'])[['R2', 'RMSE', 'MAE']].mean()
    print(summary)
    
    df_res.to_csv("train/XGBoost/result_xgboost_multitarget.csv", index=False, sep=';', decimal=',')
    print(f"\nResults saved to 'result_xgboost_multitarget.csv'")
    print(f"Best models saved in '{MODELS_DIR}/'")
else:
    print("No results generated.")