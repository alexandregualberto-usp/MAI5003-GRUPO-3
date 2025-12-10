# XGBoost Multi-Target Regression Pipeline

This project implements an end-to-end Machine Learning pipeline for shipping logistics data. It includes data preprocessing (Embeddings, Target Encoding, OHE), hyperparameter tuning using Optuna, and training optimized XGBoost models.

## Project Structure

Ensure your directories are organized as follows before running the scripts:

```text
.
├── pre_train/
│   ├── data/            
│   │   ├── df_train.parquet
│   │   ├── df_train_validation.parquet
│   │   ├── df_validation.parquet
│   │ 
│   │
│   └── processed_data/       # Generated .npz files (or downloaded from Drive)
│       ├── train_final.npz
│       ├── test_final.npz
│       └── validation_final.npz
│
├── process_data.py           # Script 1: Pre-processing
├── tune_hyperparams.py       # Script 2: Optuna Tuning
├── train.py                  # Script 3: Training Loop
└── requirements.txt          # Python dependencies
```

-----

## Setup & Installation

1.  **Clone or download this repository.**
2.  **Install the required Python packages:**

<!-- end list -->

```bash
pip install pandas numpy xgboost scikit-learn optuna
```

### Data Setup (Crucial Step)

The raw data and pre-processed numpy files are hosted on Google Drive.

**[Download Data Here](https://drive.google.com/drive/folders/1MlOavSQ-tCt83ZbyjY3Qdk1tUB7QHxQx?usp=sharing)**

1.  **Download** the contents of the Drive folder.
2.  Place the **Parquet files** inside `pre_train/data/`.
3.  *(Optional)* The Drive also contains the `processed_data` folder. If you download this, place the `.npz` files in `pre_train/processed_data/` and you can **skip Step 1** below.

-----

##  How to Run

### Step 1: Data Processing

Reads raw parquet files, applies embeddings/encoding, and saves compressed `.npz` files.

```bash
python process_data.py
```

*Output: `pre_train/processed_data/*.npz`*

### Step 2: Hyperparameter Tuning

Uses Optuna to find the best XGBoost parameters for each target variable. It validates against the validation set to prevent overfitting.

```bash
python tune_hyperparams.py
```

*Output: `hyperparams/best_params_{target}.json`*

### Step 3: Model Training

Trains the final models using 4 different feature scenarios (C1 to C4).

  * Logic: It saves the model **only** if it achieves a better RMSE than previous runs.
  * Format: Saves in native JSON format for stability.

<!-- end list -->

```bash
python train.py
```

*Output: `models_json/best_{target}.json` and `models_json/best_{target}_info.txt`*
