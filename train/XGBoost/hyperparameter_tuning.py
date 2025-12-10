import optuna
import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import sys
import json
import os
import gc

N_TRIALS = 30 
TARGETS_OF_INTEREST = [] # empty for all
DATA_DIR = "train/pre_train/processed_data" 


FILE_TRAIN = "train_final.npz"       
FILE_VAL   = "validation_final.npz"  

def get_device():
    try:
        xgb.XGBRegressor(device="cuda", n_estimators=1).fit(np.array([[1]]), np.array([0]))
        return "cuda"
    except Exception:
        return "cpu"

def load_and_prep_data():
    path_tr = os.path.join(DATA_DIR, FILE_TRAIN)
    path_val = os.path.join(DATA_DIR, FILE_VAL)

    print(f"📂 Loading data from: {DATA_DIR}")
    
    try:
        data_tr = np.load(path_tr, allow_pickle=True)
        data_val = np.load(path_val, allow_pickle=True)
    except FileNotFoundError:
        print(f"CRITICAL ERROR: Files not found in '{DATA_DIR}'.")
        print(f"Please check if {FILE_TRAIN} and {FILE_VAL} exist.")
        sys.exit(1)

    y_tr_all  = data_tr['y']
    y_val_all = data_val['y']
    target_names = data_tr['target_names']

    X_train_full = np.hstack([data_tr['X_base'], data_tr['X_merc'], data_tr['X_cont']])
    X_val_full   = np.hstack([data_val['X_base'], data_val['X_merc'], data_val['X_cont']])

    if X_train_full.shape[1] != X_val_full.shape[1]:
        diff = X_train_full.shape[1] - X_val_full.shape[1]
        if diff > 0:
            filler = np.zeros((X_val_full.shape[0], diff), dtype=X_val_full.dtype)
            X_val_full = np.hstack([X_val_full, filler])

    del data_tr, data_val
    gc.collect()

    return X_train_full, y_tr_all, X_val_full, y_val_all, target_names

def run_optimization(X_train, y_train_all, X_val, y_val_all, target_names, device):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val   = scaler.transform(X_val).astype(np.float32)

    targets_to_run = [t for t in TARGETS_OF_INTEREST if t in target_names]
    if not targets_to_run:
        targets_to_run = target_names

    os.makedirs("hyperparams", exist_ok=True)

    for target_name in targets_to_run:
        print(f"\n{'='*40}")
        print(f" TUNING TARGET: {target_name}")
        print(f"{'='*40}")

        try:
            idx = np.where(target_names == target_name)[0][0]
            y_tr_curr = y_train_all[:, idx].astype(np.float32)
            y_val_curr = y_val_all[:, idx].astype(np.float32)
        except IndexError:
            print(f"Target {target_name} not found in dataset.")
            continue

        def objective(trial):
            gc.collect()
            
            param = {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'n_estimators': 3000, 
                'device': device,
                'tree_method': 'hist',
                'n_jobs': -1,
                'max_bin': trial.suggest_categorical('max_bin', [64, 128, 256]), 
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'max_depth': trial.suggest_int('max_depth', 4, 12), 
                'subsample': trial.suggest_float('subsample', 0.5, 0.9),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 15),
                'gamma': trial.suggest_float('gamma', 0.0, 5.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            }

            model = xgb.XGBRegressor(
                **param,
                early_stopping_rounds=50,
                callbacks=[optuna.integration.XGBoostPruningCallback(trial, "validation_0-rmse")]
            )

            model.fit(
                X_train, y_tr_curr,
                eval_set=[(X_val, y_val_curr)],
                verbose=False
            )

            preds = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val_curr, preds))
            return rmse

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction='minimize', study_name=f"study_{target_name}")
        study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

        best_params = study.best_params
        final_config = best_params.copy()
        
        final_config.update({
            'n_estimators': 3000, 
            'objective': 'reg:squarederror',
            'max_bin': best_params.get('max_bin', 256)
        })
        
        save_path = f"hyperparams/best_params_{target_name}.json"
        with open(save_path, "w") as f:
            json.dump(final_config, f, indent=4)
        
        print(f"Best RMSE: {study.best_value:.4f}")
        print(f"Saved: {save_path}")

def main():
    
    device = get_device()
    X_tr, y_tr, X_val, y_val, t_names = load_and_prep_data()
    
    run_optimization(X_tr, y_tr, X_val, y_val, t_names, device)
    
    print("\n Process Finished Successfully!")

if __name__ == "__main__":
    main()
