# CHAPTER V
# RESULTS AND DISCUSSION

**Power Outage Probability Forecasting with XGBoost and SHAP-Based Feature Analysis Using Historical, Meteorological, Vegetation, and Infrastructure Reliability Data**

This chapter reports the available evidence for the Puerto Princesa City outage probability forecasting model. It follows Sections 5.1-5.8 of the thesis PDF and its research objectives: integrate the predictors, evaluate classification and probability forecasts, interpret feature contributions with SHAP, and assess engineered features. Tables and figures below are generated from the project's saved artifacts rather than the unfilled numerical placeholders in the PDF.

**Evidence and scope.** The PDF's Chapter V occupies PDF pages 47-55 (printed pages 44-52). Its example data, figure instructions, and proposed split are distinguished here from measured results. Evaluation is at the **recloser-day** level for the next local calendar day; the prediction cutoff in the implementation is 20:00 Asia/Manila on the preceding day. The supplied model is preserved. 


## 5.1 Dataset Summary

The available dataset contains **19,128 rows**, including **874 outage labels (4.57%)** and **18,254 non-outage labels (95.43%)**. The ratio of non-outage to outage labels is approximately **20.89:1**. This imbalance supports the PDF's use of precision, recall, F1-score, ROC AUC, and Brier score instead of accuracy as the primary evidence of performance.

**Table 5.1. Available dataset characteristics**

| Characteristic | Measured value or evidence limit |
| --- | --- |
| Total records | 19,128 |
| Outage records (`target_outage = 1`) | 874 |
| Non-outage records (`target_outage = 0`) | 18,254 |
| Saved predictor count | 66 original inputs |
 |
| Records containing at least one missing exported predictor | 7,487 |
| Saved `scale_pos_weight` |  |
|  |

The feature CSV contains 65 of the 66 predictor columns, the outcome label, and an additional `mapping_barangay_count` column that is not a model predictor. 

**Table 5.2. Available training, validation, and test partitions**

| Set | Records | Share (%) | Outages | Outage prevalence (%) | Verified date range |
| --- | --- | --- | --- | --- | --- |
| Training | 15935 | 83.3072 | 785 | 4.9263 | 2022-08-31 to 2026-09-21 |
| Validation | 1782 | 9.3162 | 32 | 1.7957 | 2026-02-01 to 2026-05-31 |
| Test | 1411 | 7.3766 | 57 | 4.0397 | 2026-06-02 to 2026-09-21 |

The actual record shares differ from the PDF's proposed 70%/15%/15% split. 

![Monthly outage labels in evaluation records](reports/chapter5/figures/monthly_outages.png)

*Figure 5.1. Monthly labeled outage recloser-days in the available validation and test records. Counts describe the retained sample; they are not counts of distinct city-wide outage incidents. Blue bars represent validation and orange bars represent test.*

| month | split | records | outages |
| --- | --- | --- | --- |
| 2026-02 | validation | 410 | 9 |
| 2026-03 | validation | 476 | 8 |
| 2026-04 | validation | 416 | 9 |
| 2026-05 | validation | 480 | 6 |
| 2026-06 | test | 400 | 7 |
| 2026-07 | test | 449 | 10 |
| 2026-08 | test | 389 | 27 |
| 2026-09 | test | 173 | 13 |

August has the largest outage-label count among the available evaluation months. Unequal record coverage and the partial September sample prevent this figure from establishing seasonality or the underlying city's monthly outage rate.

## 5.2 Feature Engineering Results

The saved schema specifies **66 original predictors**, grouped below. One-hot encoding and missing-value indicators expand them to 137 model input columns. Feature importance is reported after combining those transformed columns back into their corresponding original input.

**Table 5.3. Feature groups in the implemented model**

| Feature group | Inputs | Example |
| --- | --- | --- |
| Target Day Forecast Weather | 14 | forecast_precipitation_mm |
| Lagged Forecast Weather | 12 | forecast_precipitation_lag1d_mm |
| Vegetation | 6 | veg_buffer_5m |
| Outage History | 8 | outage_days_prev3d |
| Equipment History | 2 | equipment_fault_days_prev7d |
| Repair History | 2 | repair_notice_days_prev7d |
| Equipment Category History | 8 | line_damage_days_prev7d |
| Calendar | 6 | day_of_week |
| Identity | 1 | recloser |
| Wet Weather History | 3 | forecast_wet_days_prev3d |
| Weather Stress | 2 | forecast_precipitation_vs_fit_p90 |
| Weather Vegetation Interactions | 2 | forecast_gust_x_veg5m |

The implementation uses archived fixed-48-hour previous-runs **forecast weather**, rather than observed target-day weather. Lagged and rolling weather measures use daily windows. Outage and infrastructure histories use prior-day counts or recency measures; infrastructure inputs are history-based proxies rather than direct asset-condition measurements. The schema records a two-month vegetation lag and a mapping publication cutoff no later than the prediction cutoff and the frozen 2025-09-30 mapping cutoff.

Numeric inputs use training-fitted median imputation with missing indicators. Recloser and weather-code categories use training-fitted one-hot encoding. Missingness is therefore part of the model's representation and can influence its explanations. The PDF's hourly lag examples, vegetation anomaly variables, and infrastructure examples are not interchangeable with the saved daily schema. No feature-selection log is supplied, so this report does not attribute absent variables to correlation filtering or another unrecorded removal rule.

## 5.3 Model Development and Hyperparameter Tuning

The supplied artifact is an XGBoost binary probability model with the following saved configuration.

**Table 5.4. Saved XGBoost configuration**

| Parameter | Saved value |
| --- | --- |
| objective | binary:logistic |
| tree_method | hist |
| max_depth | 10 |
| eta | 0.05 |
| min_child_weight | 1 |
| lambda | 1.0 |
| scale_pos_weight | 1 |
| max_delta_step | 1 |
| subsample | 0.9 |
| colsample_bytree | 0.9 |
| max_bin | 512 |
| seed | 42 |
| nthread | 1 |
| boosting rounds | 400 |

The saved `scale_pos_weight` is 1; it is not the negative-to-positive class ratio suggested as a placeholder in the PDF. The package records 400 boosting rounds, but does not include a hyperparameter search history, search ranges, tuning objective, or early-stopping log. These procedures cannot be reconstructed as findings from the current evidence.

The saved probability calibration is a sigmoid transformation of the raw model probability:

```text
calibrated_probability = sigmoid(1.1343608862447834 * logit(raw_probability)
                                + 0.6740749548179885)
```

The raw warning threshold is 0.1588377059; its calibrated threshold is **0.2285013004**, equivalent to **22.850130%**. Predictions at or above the calibrated threshold trigger a warning. Low risk is below 5%; Medium is from 5% to below the warning threshold; High is at or above the threshold. The runtime uses the full saved threshold precision. These are bins of a binary probability output rather than separately trained risk classes.

The saved configuration states that test data were not used to choose the threshold. The precise original threshold-selection objective is not archived. No threshold or hyperparameter was changed to produce the main results in this chapter.

## 5.4 Predictive Performance of the Model (Objective 2)

Metrics were recomputed from the saved labels, percentages, and verified warning decisions, and checked against `results.json`. At the exact decision boundary, converting a saved percentage back to probability can shift the value by one floating-point unit; classification metrics preserve the saved warning after checking threshold consistency.

**Table 5.5. Reproduced performance of the archived selected model**

| Metric | Validation | Test |
| --- | --- | --- |
| precision | 0.641026 | 0.620253 |
| recall | 0.781250 | 0.859649 |
| F1 | 0.704225 | 0.720588 |
| ROC_AUC | 0.946018 | 0.985566 |
| average_precision | 0.710665 | 0.821081 |
| Brier_score | 0.008438 | 0.014951 |

On the test records, the model identified **49 of 57 labeled outage recloser-days**, producing **85.96% recall**. It issued **79 warnings**, of which **49 corresponded to positive labels**, giving **62.03% precision**. Its F1-score was **0.720588**, and ROC AUC was **0.985566**. Average precision, included to describe ranking performance under imbalance, was **0.821081**.

The saved model exceeds the PDF's proposed F1 benchmark of 0.60 and ROC AUC benchmark of 0.75 on both evaluation sets. Its Brier scores are below the PDF's proposed 0.25 cutoff. That cutoff alone is weak evidence under low outage prevalence; Section 5.5 provides a prevalence-based baseline and reliability diagram.

![Validation and test ROC curves](reports/chapter5/figures/roc_curve.png)

*Figure 5.2. ROC curves from saved calibrated probabilities. ROC AUC measures ranking across thresholds; it does not establish calibration or a suitable operational warning threshold.*

**Table 5.6. Test confusion matrix at calibrated threshold 0.2285013004**

| Actual label | Predicted: No warning | Predicted: Warning |
| --- | ---: | ---: |
| Non-outage | TN = 1,324 | FP = 30 |
| Outage | FN = 8 | TP = 49 |

The model missed 8 positive labels and generated 30 warnings against negative labels. This is consistent with the study's concern for missed outages, while showing a material false-warning burden. Because labels derive from available interruption records or advisories, a negative label means no matching recorded interruption; it is not independently verified continuous electricity service.

### All-Input Retraining on the Preserved Partition

The experimental model F was trained on all 15,935 saved training records. Its validation-selected threshold is 0.2882164906. On test it obtained **precision 67.65%, recall 80.70%, F1 0.736000, ROC AUC 0.986784, and Brier score 0.013860**. It detected 46 positive labels, missed 11, and generated 22 false warnings. This is a new fit on the existing partitions, not an independently verified chronological evaluation.

| split | precision | recall | F1 | ROC_AUC | average_precision | Brier_score |
| --- | --- | --- | --- | --- | --- | --- |
| validation | 0.696970 | 0.718750 | 0.707692 | 0.945750 | 0.703049 | 0.008446 |
| test | 0.676471 | 0.807018 | 0.736000 | 0.986784 | 0.849911 | 0.013860 |

For the PDF's proposed benchmarks, experimental test F1 **meets 0.60**, ROC AUC **meets 0.75**, and Brier score **is below 0.25**. The Brier cutoff alone is weak evidence under imbalance. Differences from the archived model can reflect uniform weights, refitted preprocessing and weather-stress references, and the documented calibration and threshold-selection procedure. The original training recipe is not fully archived.


## 5.5 Probability Calibration (Objective 2)

The following analysis compares uncalibrated probabilities with the saved sigmoid-calibrated probabilities. The calibration was fitted on validation, so validation calibration results are in-sample for that fitting step.

**Table 5.7. Probability error and reference baselines**

| Set | Raw Brier | Calibrated Brier | Training-rate baseline Brier | Brier skill |
| --- | --- | --- | --- | --- |
| validation | 0.009160 | 0.008438 | 0.018615 | 0.546721 |
| test | 0.015807 | 0.014951 | 0.038844 | 0.615100 |





![Validation and test reliability diagrams](reports/chapter5/figures/reliability_diagram.png)

*Figure 5.3. Reliability diagrams using ten equal-width probability bins. Point labels show bin sample sizes. Points above the diagonal indicate underprediction; points below it indicate overprediction. Sparse bins have substantial sampling uncertainty.*

The mean test probability is **4.4851%**, compared with an observed positive-label rate of **4.0397%**. Agreement of aggregate means does not prove reliability within risk ranges. Most observations lie in the lowest probability bin, while the higher bins contain relatively few records. 
## 5.6 SHAP-Based Feature Importance (Objective 3)

### 5.6.1 Global Feature Importance

Exact tree-path-dependent TreeSHAP contributions were computed for the **archived selected model** using XGBoost's `pred_contribs=True` on all **1,411 test records**. These explanations are not explanations of the newly retrained experimental models. Contributions explain the frozen model's **raw log-odds margin**, following the [XGBoost API definition](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.Booster.predict), rather than percentage-point changes in calibrated probability. One-hot columns and missing indicators were summed within each original feature for every record before taking the mean absolute value. The contribution sum, including the bias term, was checked against the raw model margin. No test background dataset was used to fit or change the model.

![Top 15 original predictors by mean absolute SHAP value](reports/chapter5/figures/shap_importance.png)

*Figure 5.4. Test-set global importance. Missing-indicator effects are included in their original predictor's attribution.*

**Table 5.8. Ten highest-ranked original predictors**

| Rank | feature | group | mean_abs_SHAP | mean_signed_SHAP |
| --- | --- | --- | --- | --- |
| 1 | day_of_year_cos | calendar | 0.391193 | -0.386781 |
| 2 | forecast_wind_speed_max_kmh | target_day_forecast_weather | 0.383656 | 0.058689 |
| 3 | outage_days_prev90d | outage_history | 0.345147 | -0.342517 |
| 4 | days_since_last_known_outage | outage_history | 0.332283 | -0.188285 |
| 5 | day_of_week | calendar | 0.282568 | -0.043084 |
| 6 | forecast_wind_direction_cos | target_day_forecast_weather | 0.263955 | -0.184078 |
| 7 | recloser | identity | 0.239752 | -0.180289 |
| 8 | forecast_gust_x_veg5m | weather_vegetation_interactions | 0.211723 | -0.167005 |
| 9 | forecast_wind_gust_max_prev3d_kmh | lagged_forecast_weather | 0.199301 | -0.058999 |
| 10 | forecast_precipitation_sum_prev7d_mm | lagged_forecast_weather | 0.196364 | -0.046498 |

The largest mean absolute attribution belongs to **`day_of_year_cos`**. The next two are **`forecast_wind_speed_max_kmh`** and **`outage_days_prev90d`**. These rankings describe how the fitted model uses its inputs on this test sample. They do not establish causal outage drivers. Mean signed SHAP describes an average contribution relative to the model's reference margin, not a monotonic effect of increasing a feature.

![SHAP distribution by feature](reports/chapter5/figures/shap_beeswarm.png)

*Figure 5.5. SHAP distribution plot. Color indicates relative numeric feature value within the 5th-95th percentile range; categorical and missing values are gray. Horizontal position indicates positive or negative contribution to raw log-odds.*

### 5.6.2 Contribution by Feature Group

**Table 5.9. Aggregated feature-group attribution**

| group | mean_abs_SHAP | share_percent |
| --- | --- | --- |
| target_day_forecast_weather | 1.927769 | 30.264311 |
| lagged_forecast_weather | 1.396361 | 21.921660 |
| outage_history | 0.954199 | 14.980103 |
| calendar | 0.861606 | 13.526466 |
| vegetation | 0.498474 | 7.825608 |
| weather_vegetation_interactions | 0.383882 | 6.026614 |
| identity | 0.239752 | 3.763902 |
| wet_weather_history | 0.034621 | 0.543518 |
| equipment_history | 0.034418 | 0.540336 |
| weather_stress | 0.026971 | 0.423414 |
| equipment_category_history | 0.011592 | 0.181987 |
| repair_history | 0.000133 | 0.002082 |

**Target Day Forecast Weather** has the largest sum of feature-level mean absolute SHAP values, accounting for **30.26%** of that total. Group shares sum mean absolute original-feature attributions; they are not fractions of predictive accuracy, causal contribution, or the mean absolute value of a net group effect. Correlated predictors can share or substitute for attribution.

### 5.6.3 Feature-Level Interpretation

![SHAP dependence for precipitation, gust, and vegetation inputs](reports/chapter5/figures/shap_dependence.png)

*Figure 5.6. Dependence plots for precipitation, wind gust, and mean barangay NDVI. Only nonmissing values are plotted; missingness effects remain included in global importance. Plots show associations in the fitted model rather than physical failure thresholds.*

These plots provide the meteorological and vegetation views requested in the PDF. Weather, vegetation, equipment histories, calendar effects, and recloser identity should be interpreted together because the model can represent interactions. A strong attribution from an infrastructure-history feature indicates use of recent reported faults, not a measured condition or age of a physical asset. This report does not infer a universal rainfall, wind, or NDVI trigger from the displayed associations.

## 5.7 Contribution of Lagged and Cumulative Outage Features (Objective 4)

### Retraining Ablation

The explicit metadata sidecar was joined by dataset row number and checked against the calendar and saved evaluation identities. 
Each experimental model uses the saved XGBoost hyperparameters and 400 rounds, with preprocessing fitted on the complete saved training partition. Weather-stress reference percentiles are also refitted on that partition. Each model receives validation-fitted sigmoid calibration and a maximum-F1 threshold selected on validation, then frozen for test scoring. Training uses uniform weights because original sample weights are unavailable. No test score is used to choose a variant for deployment; all prespecified variants are reported. These are new experiments, not a reconstruction of the undocumented original tuning process.


**Table 5.10. Retraining ablations on the preserved test partition**

| variant | features | precision | recall | F1 | ROC_AUC | Brier_score | threshold |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A. Weather + calendar + identity | 21 | 0.602564 | 0.824561 | 0.696296 | 0.973192 | 0.019468 | 0.373692 |
| B. A + lagged/wet weather | 36 | 0.734375 | 0.824561 | 0.776860 | 0.971430 | 0.018995 | 0.496778 |
| C. A + outage history | 29 | 0.671642 | 0.789474 | 0.725806 | 0.977649 | 0.018192 | 0.265383 |
| D. A + vegetation/interactions | 29 | 0.761905 | 0.842105 | 0.800000 | 0.987794 | 0.011338 | 0.336207 |
| E. A + infrastructure history | 33 | 0.666667 | 0.771930 | 0.715447 | 0.976548 | 0.017265 | 0.357663 |
| F. All inputs | 66 | 0.676471 | 0.807018 | 0.736000 | 0.986784 | 0.013860 | 0.288216 |

**Validation results and frozen thresholds**

| variant | precision | recall | F1 | ROC_AUC | Brier_score | threshold |
| --- | --- | --- | --- | --- | --- | --- |
| A. Weather + calendar + identity | 0.826087 | 0.593750 | 0.690909 | 0.947143 | 0.008414 | 0.373692 |
| B. A + lagged/wet weather | 1.000000 | 0.562500 | 0.720000 | 0.961964 | 0.008114 | 0.496778 |
| C. A + outage history | 0.611111 | 0.687500 | 0.647059 | 0.922714 | 0.010193 | 0.265383 |
| D. A + vegetation/interactions | 0.794118 | 0.843750 | 0.818182 | 0.959339 | 0.006382 | 0.336207 |
| E. A + infrastructure history | 0.730769 | 0.593750 | 0.655172 | 0.943821 | 0.009115 | 0.357663 |
| F. All inputs | 0.696970 | 0.718750 | 0.707692 | 0.945750 | 0.008446 | 0.288216 |


### Available Feature-Sensitivity Analysis

As an additional analysis, feature groups were jointly permuted across test records with a fixed random seed and repeated measurements. The saved model, calibration, and decision threshold stayed fixed. Features within each group were moved together using the same row permutation. The group definitions and every repeat's results are saved alongside the report.

**Table 5.11. Repeated test-set permutation sensitivity**

| Feature group | F1, mean (SD) | ROC AUC, mean (SD) | Brier, mean (SD) | F1 drop | AUC drop | Brier increase |
| --- | --- | --- | --- | --- | --- | --- |
| Infrastructure history | 0.709396 (0.003279) | 0.985970 (0.000384) | 0.015121 (0.000209) | 0.011192 | -0.000404 | 0.000170 |
| Lagged forecast weather | 0.276749 (0.054500) | 0.913794 (0.008823) | 0.032579 (0.001138) | 0.443840 | 0.071772 | 0.017629 |
| Outage history | 0.575278 (0.030973) | 0.964855 (0.001024) | 0.022945 (0.000967) | 0.145310 | 0.020711 | 0.007994 |
| Vegetation and interactions | 0.409713 (0.040116) | 0.939532 (0.007831) | 0.031464 (0.002252) | 0.310875 | 0.046034 | 0.016513 |

F1 drop and AUC drop equal unperturbed performance minus the mean perturbed score. Brier increase equals the mean perturbed Brier score minus the unperturbed value. Positive changes indicate worse performance after perturbation; a negative change indicates improvement. Standard deviations describe variability across permutations, not confidence intervals for future outages. The CSV includes additional precision and recall summaries.

The largest mean ROC AUC decrease in these experiments occurs for **Lagged forecast weather** (0.071772). This tests the archived model's sensitivity to disrupting inputs. It **does not measure the benefit of adding that feature group during training** and is separate from the retraining comparisons. Permutation can also break relationships with unpermuted interaction variables or create unrealistic combinations; correlated features can conceal or redistribute apparent importance.

## 5.8 Discussion

### Relation to the Reviewed Literature

The implemented model follows the thesis's broad approach of combining historical interruption information with weather, vegetation, temporal, and infrastructure-history predictors. Chapter V proposes comparison with Saidi et al. (2021), Putra et al. (2025), Lee et al. (2025), Wang et al. (2022), and Talukdar and Toweh (2024). No numerical superiority claim is made here: studies with different regions, prediction horizons, event definitions, and class balances do not provide a matched benchmark for this test sample.

### Weather, History, Vegetation, and Infrastructure

The computed SHAP results establish which retained predictors the frozen model uses, while permutation sensitivity measures how its test outputs change when selected information is disrupted. Both are useful evidence of model behavior. Establishing improvement from combining sources requires the retraining comparisons described in Section 5.7. Vegetation resolution, missing imagery, frozen spatial mappings, and history-based equipment proxies limit the physical conclusions that can be drawn from model attribution.

### Routine and Weather-Driven Outages

The saved prediction tables do not classify each interruption by cause or identify a validated severe-weather regime. Monthly label counts alone cannot establish separate performance during ordinary operation, heavy rain, or typhoons. Such claims remain outside the measured results in this chapter.

### Practical Implications for PALECO

The model's test recall and ranking performance indicate potential value for prioritizing recloser-days for further review. The 30 false warnings and 8 missed positive labels show why a prospective pilot should evaluate warning costs and response capacity. The thesis is model-centric; the current evidence does not demonstrate crew deployment benefits, outage prevention, or a production forecasting system.

### Evidence Limits and Remaining Work

The test set contains 57 positive recloser-day labels across 104 represented dates. 

Objectives 1 and 2 are supported by the preserved model, feature schema, replay checks, and recomputed metrics, within these limits. Objective 3 now has measured SHAP evidence. Objective 4 now has a measured sensitivity analysis and completed retraining comparisons under the documented experimental protocol.

