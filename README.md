# Detecting Deception from Speech

Can acoustic features alone tell a true story from a made-up one? An end-to-end audio
classification pipeline on the public
[MLEnd Deception dataset](https://github.com/MLEndDatasets/Deception): roughly 100 two-minute
recordings of narrated stories in several languages, each labelled true or deceptive.

**Notebook:** [`speech_deception_detection.ipynb`](speech_deception_detection.ipynb)

## Pipeline

1. **Split first:** stratified 60 / 20 / 20 train / validation / test split at the recording level,
   with an explicit overlap check. An early version scored a suspicious 100% AUC, which turned out
   to be leakage between sets, so the check stayed in.
2. **Segment:** every recording is cut into 30-second segments; shorter clips are dropped.
3. **Features** (librosa): MFCC means and variances, pitch, power and spectral statistics, scaled
   with `StandardScaler`.
4. **Models:** SVM, Random Forest and XGBoost compared on validation AUC, accuracy, precision,
   recall and F1.
5. **Feature selection:** Random Forest importances with a 3% threshold (34 → 15 features).
6. **Tuning:** Optuna search over Random Forest hyperparameters, maximising validation AUC.
7. **Error analysis:** misclassified segments by language and label.

## Results

| Stage | AUC | Accuracy | F1 |
|---|---|---|---|
| Random Forest, validation (untuned) | 0.55 | 0.51 | 0.49 |
| Tuned Random Forest, validation | **0.72** | 0.70 | 0.63 |
| Tuned Random Forest, **test** | 0.44 | 0.51 | 0.55 |

The honest conclusion is that the tuned model does **not** generalise: the validation gain comes
from tuning on a very small validation set and disappears on the test set. With about 100
speakers, many languages and segment-level features, acoustic cues alone are too weak and too
speaker-dependent for reliable deception detection. Speaker-grouped cross-validation, more data
and prosodic or linguistic features would be the next steps.

## Run it

```bash
pip install -r requirements.txt
jupyter notebook speech_deception_detection.ipynb
```

Download the MLEnd Deception audio and metadata, then set the two paths in the data-loading cell.
