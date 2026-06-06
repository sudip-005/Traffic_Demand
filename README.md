# Traffic Demand Forecasting

Short overview

This repository contains data, notebooks, and a training script for exploring and modeling traffic demand. Use the notebooks for EDA and feature engineering, and the training script to build a production-style model and produce submission files.

## Repository structure

- [feature_metadata.csv](feature_metadata.csv): metadata about features
- [model_feature_list.csv](model_feature_list.csv): selected model features
- [sample_submission.csv](sample_submission.csv): example submission format
- [submission.csv](submission.csv): produced submission (if any)
- [test_processed.csv](test_processed.csv): processed test set ready for modeling
- [test.csv](test.csv): raw test data
- [train_high_performance_model.py](train_high_performance_model.py): training script to build and finalise a model
- [train_processed.csv](train_processed.csv): processed training set ready for modeling
- [train.csv](train.csv): raw training data
- [traffic_demand.ipynb](traffic_demand.ipynb): notebook for demand-specific analysis and modeling
- [traffic.ipynb](traffic.ipynb): exploratory notebooks and preprocessing steps

## Quickstart

1. Create a Python environment and install common packages used in the notebooks and scripts:

```powershell
python -m venv venv
venv\Scripts\activate
pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn jupyter
```

2. Open the notebooks for exploration:

- Open [traffic_demand.ipynb](traffic_demand.ipynb) to follow the analysis and model-building steps.
- Open [traffic.ipynb](traffic.ipynb) for preprocessing and EDA.

3. Train the model / generate a submission

- If you prefer the scriptable route, run the training script (it expects processed CSVs in the repository):

```powershell
python train_high_performance_model.py
```

- Otherwise, run the relevant notebook cells to reproduce preprocessing, training, and prediction steps. The output submission should follow the format in [sample_submission.csv](sample_submission.csv).

## Notes

- If `*_processed.csv` files are missing or out-of-date, run the preprocessing cells in the notebooks to regenerate them.
- Inspect `feature_metadata.csv` and `model_feature_list.csv` to understand which features were engineered and selected for modeling.

## Next steps

- Add a `requirements.txt` if you want pinned dependencies.
- Add a brief `USAGE.md` or CLI docs for `train_high_performance_model.py` if it supports arguments.

## Contact

If you need help reproducing results or adding packaging, open an issue or ask the repository owner.
