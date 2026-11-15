# Missed Signals

8th Annual Datathon for Social Good, Advanced Track.

## The problem we're solving

**Hospitals often spot sepsis too late.** Sepsis is a deadly reaction to an infection, and every hour without treatment matters. To catch it early, hospitals use a checklist score that sets off an alarm when a patient looks sick.

**At its emergency level, that alarm gives no early warning for about half of sepsis patients.** And when the finger clip that measures oxygen reads too high, which happens more often on darker skin, it misses even more.

## Our solution

**A model that warns earlier, with the same number of alarms.** It watches how a patient's vital signs change over time, instead of looking at one moment.

On 40,000 real ICU stays, the checklist warned early for **49%** of sepsis patients. Our model warned early for **67%**. The bad finger clip did not fool it.

![Patients warned early: checklist vs our model](figures/fig2_caught.png)

## Key words

| Word | Meaning |
|---|---|
| **Sepsis** | The body's extreme reaction to an infection. It damages organs and can kill within hours. |
| **ICU** | Intensive care unit, where the sickest patients stay. |
| **Checklist score (NEWS2)** | What hospitals use today. A nurse gives points for heart rate, breathing, oxygen, blood pressure, temperature and alertness. A high total sets off an alarm. |
| **Oxygen clip** | The finger clip that measures oxygen. It can read too high, more often on darker skin. |
| **Model** | A computer program that learns patterns from past patients. |
| **Warned early** | An alarm went off in the 12 hours before sepsis started. |

## Why the checklist misses patients

It has two blind spots:

1. **It sees one moment.** It can't tell that a heart rate has climbed for 6 hours.
2. **It trusts the oxygen clip.** If the clip reads too high, a sick patient looks fine.

## What we did

1. **Checked the datathon data.** Its labels are the checklist score itself, and the patient ID leaked the answer. It has no time, so we used it only to check our code.
2. **Added real ICU data.** 40,336 stays with vital signs every hour and the hour each patient got sepsis.
3. **Built a model** that also looks at how each vital sign changed over the last 6 hours.
4. **Made the test fair.** We trained on one hospital and tested on another. The model got the same number of alarms as the checklist.
5. **Tested a bad oxygen clip.** We made every oxygen reading 3 points too high and ran the test again.

## What we found

| | Checklist | Our model |
|---|---|---|
| Sepsis patients warned early | 49% | **67%** |
| ...when the oxygen clip reads too high* | 78% → 73% | 84% → 86% |
| False alarms per patient per day | 1.21 | 1.24 |

\* At the lower ("urgent") alarm level.

- **141 more patients warned early**, out of 788.
- **A bad oxygen clip cost the checklist about 42 patients.** Our model lost none.
- **It worked for women, men, younger and older patients.** It was not better in medical ICUs (81% vs 82%).

## Limits

- **The model's biggest clue is time spent in the ICU.** Without it, the model only matches the checklist.
- **We simulated the oxygen-clip error**, because no dataset records race.
- **ICU patients and sepsis only.** This is a prototype, not a medical tool.

---

<details>
<summary><b>Technical details</b> (methods, full results, how to run)</summary>

### Full results, hospital B (788 sepsis patients)

| Measure | NEWS2 | Logistic regression | LightGBM | LightGBM, no ICU time |
|---|---|---|---|---|
| Hourly AUROC | 0.701 | 0.757 | 0.784 | 0.725 |
| Caught, alarm budget = NEWS2 >= 7 | 49.1% | 60.5% | 67.0% | 50.9% |
| False alarms per patient-day (>= 7) | 1.21 | 1.01 | 1.24 | 1.14 |
| Caught, alarm budget = NEWS2 >= 5 | 78.2% | 81.6% | 83.8% | 75.4% |
| False alarms per patient-day (>= 5) | 4.18 | 3.93 | 4.80 | 3.99 |
| Caught (>= 5) with SpO2 + 3 | 72.8% | 81.3% | 85.9% | |

Health Risk dataset: our NEWS2 code matches 987 of 1,000 labels; `Patient_ID` alone gives 99.7% accuracy (dropped); SpO2 + 3 moves 155 of 1,000 patients to a lower band.

### Advanced Track checklist

| Requirement | Where in `notebook.ipynb` |
|---|---|
| Trained model | Section 5: logistic regression and LightGBM, trained on hospital A only |
| Baseline | NEWS2 (published clinical rule) and majority class |
| Held-out evaluation | Section 2: 30% held-out split. Section 6: hospital B, never used for training or tuning |
| Error and fairness analysis | Sections 2a and 7: oxygen scale, oxygen use, consciousness, sex, age group, ICU type. Section 8: missed patients |
| Data decisions, checked by hand | Sections 1 and 4. `news2.py` is unit-tested against hand-scored rows |
| Reproducibility | Fixed seed 42, `requirements.txt`, raw data to results in one notebook |

### Method

1. **NEWS2** (`news2.py`): Royal College of Physicians scoring, SpO2 scales 1 and 2.
2. **Features**: 5 shared vital signs carried forward; 6-hour max, min and change; oxygen use (FiO2 > 0.21); age; sex; hours in ICU (ICULOS). Consciousness is not recorded, so it is scored as alert.
3. **Alarm-budget matching**: thresholds set on hospital A to match the NEWS2 >= 5 or >= 7 alarm rate, then applied unchanged to hospital B.
4. **Early warning**: caught = any alarm in the 12 hours before onset (first `SepsisLabel` hour + 6). Only stays with 12+ hours before onset are scored.

### How to run

```bash
pip install -r requirements.txt
# data/Health_Risk_Dataset.csv            (datathon Drive folder)
# data/physionet/training_setA/*.psv      (PhysioNet 2019 Challenge, training_setA)
# data/physionet/training_setB/*.psv      (PhysioNet 2019 Challenge, training_setB)
jupyter nbconvert --to notebook --execute notebook.ipynb
```

More charts are in `figures/`. Oxygen-clip study: [Sjoding et al., NEJM 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7808260).

</details>

## Data

- **Health_Risk_Dataset.csv**: from the datathon (Apache 2.0).
- **ICU data**: PhysioNet/Computing in Cardiology Challenge 2019 (Reyna et al., Critical Care Medicine 2020; CC BY 4.0, https://doi.org/10.13026/v64v-d857). We downloaded it from the Kaggle copy [Prediction of Sepsis](https://www.kaggle.com/datasets/salikhussaini49/prediction-of-sepsis), which lists CC BY-NC-SA 4.0.
