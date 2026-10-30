# Day-Ahead XGBoost Outage Forecast: Selected Model Report

## Purpose and selected result

It estimates next-day outage probability per recloser and outputs a percentage, binary warning and Low/Medium/High risk band.

| Metric | Validation | Test |
|---|---:|---:|
| Precision | 64.10% | 62.03% |
| Recall | 78.12% | 85.96% |
| F1-score | 70.42% | 72.06% |
| ROC AUC | 0.946018 | 0.985566 |
| Average precision | 0.710665 | 0.821081 |
| Brier score | 0.008438 | 0.014951 |

| Classification outcome | Validation | Test |
|---|---:|---:|
| Detected outages (TP) | 25 | 49 |
| False warnings (FP) | 14 | 30 |
| Missed outages (FN) | 7 | 8 |
| Correctly unwarned non-outages (TN) | 1,736 | 1,324 |

Test contained 57 labeled outage records and 1,354 labeled non-outage records. Of 79 warnings, 49 corresponded to labeled outages: precision 62.03%. The model detected 49 of 57 labeled outages: recall 85.96%. 

## Data and replacement method

record training dates span August 2022-January 2026. Validation is February-May 2026 (1,782 records, 32 outages); test is June-September 2026 (1,411 records, 57 outages).

The starting training group had 14,224 original-record rows (717 outage / 13,507 non-outage).



## Model and probability outputs

The model uses 66 inputs: archived forecast weather and rolling summaries, vegetation, prior outage/equipment histories, calendar, recloser identity and weather-vegetation interactions. The exact lis are in [feature_schema.json](feature_schema.json). Numeric median imputation with missing indicators and categorical one-hot encoding are fitted on training. Generation flags, targets, parent IDs and weights are not predictor features.

XGBoost parameters: binary:logistic objective, hist tree method, depth 10, 400 rounds, learning rate 0.05, min_child_weight 1, L2 regularization 1, scale_pos_weight 1, max_delta_step 1, row and column subsampling 0.9, max_bin 512, seed 42, one thread. Exact parameters and sigmoid calibration are in [model_configuration.json](model_configuration.json).

The raw warning threshold is 0.1588377059. After validation-fitted sigmoid calibration, the threshold is 0.2285013004. Outage percentage equals calibrated probability times 100. Low risk is below 5%; Medium is from 5% to below 22.850130%; High is at least 22.850130%, using the full-precision saved threshold. High triggers a warning. These are bins of a binary probability, not separately learned three-class outcomes.

## Evaluation limits and calibration discussion

Each candidate threshold was set on validation and frozen before test scoring within the source experiment.


Validation Brier is 0.008438 and test Brier is 0.014951. Calibration was fitted on validation, making its validation Brier in-sample. Aggregate Brier scores under class imbalance.

## Loading and prediction

```powershell
python -m pip install -r requirements.txt
python predict.py --input prepared_features.csv --output predictions.csv
```

Prepared inputs must contain exactly the feature  andclisted in feature_schema.json. The runtime checks calendar consistency. Predictions are saved in [validation_predictions.csv](predictions/validation_predictions.csv) and [test_predictions.csv](predictions/test_predictions.csv).

The model is preserved without retraining. Verification checks the saved hash, exact probability/warning replay, dataset accounting and replacement lineage; see verification.json.

## Combined dataset

[complete_model_dataset.csv](datasets/complete_model_dataset.csv). The dataset_split column identifies training, validation or test. It contains 19,128 rows: 15,935 training, 1,782 validation and 1,411 test. All panel IDs are unique. Each partition was read back and checked against its source CSV without feature or label changes.
    
