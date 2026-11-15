# Missed Signals

**Hospital warning scores miss patients who are getting sicker, and a pulse-oximeter error hides even more of them. We tested by how much, on 40,000 real ICU stays.**

8th Annual Datathon for Social Good, Advanced Track.

## Results

| Finding | Number |
|---|---|
| The datathon Health Risk labels are NEWS2 itself | 987 of 1,000 rows match our NEWS2 code. All 13 differences are one scale-2 rule |
| `Patient_ID` leaks the label | 99.7% accuracy from the ID alone. We drop it |
| Sepsis patients caught in the 12 h before onset, at the NEWS2 emergency alarm level (hospital B, 788 patients, same number of alarms) | NEWS2 49%; model with vital-sign trends 51%; model with trends + time in ICU **67%** |
| False alarms per patient-day at that level | NEWS2 1.21; trends + ICU time 1.24; logistic regression 1.01 (catches 61%) |
| Pulse oximeter reads 3 points high (hospital B) | NEWS2 catches 78% to 73% (about 42 fewer of 788 patients). Trained models: unchanged |
| Pulse oximeter reads 3 points high (Health Risk data) | 155 of 1,000 patients move to a lower risk band |

Hourly AUROC on hospital B: NEWS2 0.701, logistic regression 0.757, LightGBM 0.784, LightGBM without ICU time 0.725.

## Advanced Track checklist

| Requirement | Where in `notebook.ipynb` |
|---|---|
| Trained model | Section 5: logistic regression and LightGBM, trained on hospital A only |
| Baseline | NEWS2 (published clinical rule) and majority class |
| Held-out evaluation | Section 2: 30% held-out split. Section 6: hospital B, never used for training or tuning |
| Error and fairness analysis | Sections 2a and 7: oxygen scale, oxygen use, consciousness, sex, age group, ICU type. Section 8: missed patients |
| Data decisions, checked by hand | Sections 1 and 4. `news2.py` is unit-tested against hand-scored rows |
| Reproducibility | Fixed seed 42, `requirements.txt`, raw data to results in one notebook |

## How to run

```bash
pip install -r requirements.txt
# data/Health_Risk_Dataset.csv            (datathon Drive folder)
# data/physionet/training_setA/*.psv      (PhysioNet 2019 Challenge, training_setA)
# data/physionet/training_setB/*.psv      (PhysioNet 2019 Challenge, training_setB)
jupyter nbconvert --to notebook --execute notebook.ipynb
```

Runtime: about 3 minutes on a laptop.

## Method

1. **NEWS2** (`news2.py`): Royal College of Physicians scoring for respiratory rate, SpO2 (scales 1 and 2), oxygen use, systolic BP, heart rate, consciousness and temperature.
2. **PhysioNet features**: the 5 vitals both datasets share, carried forward within each stay; 6-hour max, min and change for each; oxygen use (FiO2 > 0.21); age; sex; hours in ICU.
3. **Fair comparison**: each model's alarm threshold is set on hospital A so it raises the same share of hourly alarms as NEWS2 (>= 5 or >= 7). Thresholds are then applied unchanged to hospital B.
4. **Early warning**: a sepsis patient counts as caught if any alarm fires in the 12 hours before onset (onset = first `SepsisLabel` hour + 6, per the challenge definition). Only stays with at least 12 hours before onset are scored.

## Limitations and bias

- **Pulse-oximeter bias.** Oximeters can overestimate oxygen on darker skin; in one large study, Black patients had nearly 3 times the rate of low oxygen that the oximeter missed ([Sjoding et al., NEJM 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7808260)). Neither dataset records race, so we **simulate** the error. We cannot measure it by race.
- **Time in ICU drives most of the model's gain.** A model on vital signs alone only matches NEWS2. Time in ICU is information staff have, but in this dataset it may also reflect how records were collected. We report both models.
- **ICU, not ward patients.** PhysioNet covers ICU stays at two US hospital systems. Ward patients are less sick and measured less often.
- **One outcome.** We test sepsis only.
- **Health Risk labels are a rule.** That dataset cannot show where NEWS2 is wrong, so we use it to validate our NEWS2 code and the bias test.
- **Small subgroups.** In the Health Risk test set only 31 patients are not alert; LightGBM's macro-F1 drops to 0.63 there.
- **Prototype only.** Not for clinical use. Real use needs prospective testing, data with race and arterial blood gas readings, and clinical review.

## Data and licenses

- Health_Risk_Dataset.csv: datathon-provided, Apache 2.0.
- PhysioNet/Computing in Cardiology Challenge 2019 (Reyna et al., Critical Care Medicine 2020), CC BY 4.0. https://doi.org/10.13026/v64v-d857
