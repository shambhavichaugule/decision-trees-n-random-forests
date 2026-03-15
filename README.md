# Decision Trees & Random Forest — Predicting Shipment Delays

## Project Summary

Built and compared Decision Tree and Random Forest classifiers to predict shipment delays using a real-world Nigerian retail and e-commerce supply chain dataset from Hugging Face. This project is the third in a series — using the same dataset as the logistic regression project to enable direct model comparison.

**Dataset:** [electricsheepafrica/nigerian_retail_and_ecommerce_supply_chain_logistics_data](https://huggingface.co/datasets/electricsheepafrica/nigerian_retail_and_ecommerce_supply_chain_logistics_data)

**Business Problem:** Can we predict whether a shipment will be delayed before it ships — so logistics teams can intervene early?

**Previous Project:** [Logistic Regression — Shipment Delay Prediction](https://github.com/shambhavichaugule/logistic-regression-project)

---

## What I Built

Two classification models predicting:
- `0` → Shipment arrives on time (days_late ≤ 2)
- `1` → Shipment is delayed (days_late > 2)

---

## Key Learnings

### 1. What is a Decision Tree?

A decision tree learns a series of yes/no questions from data to arrive at a prediction — exactly like a flowchart.

```
Is shipping_cost > 5000?
├── Yes → Is quantity > 50?
│         ├── Yes → DELAYED
│         └── No  → ON TIME
└── No  → Is destination_city == Lagos?
          ├── Yes → DELAYED
          └── No  → ON TIME
```

**Advantages:**
- Highly interpretable — you can see exactly why a prediction was made
- Handles non-linear relationships
- No need to scale features
- Handles multiple classes natively

**Disadvantages:**
- Prone to overfitting without depth control
- Unstable — small data changes can produce very different trees

### 2. What is a Random Forest?

A random forest builds hundreds of decision trees, each trained on a slightly different random subset of the data, then takes a majority vote across all trees.

```
Tree 1   → delayed
Tree 2   → on time
Tree 3   → delayed
Tree 4   → delayed
Tree 5   → on time
...
Tree 100 → delayed

Majority vote → delayed ✅
```

**Why it's better than a single tree:**
- Reduces overfitting by averaging many trees
- More stable — not sensitive to small data changes
- Generally higher accuracy than a single decision tree

### 3. Overfitting — The Most Important Discovery

Running the decision tree with no depth limit:

```
Training Accuracy: 100%
Testing Accuracy:   53%
Gap:               47%
```

The model memorized every training example perfectly but failed completely on new data. This is **overfitting** — like a student who memorizes answers word for word but can't answer any new questions.

**Fix — Limit tree depth:**
```python
dt_model = DecisionTreeClassifier(max_depth=10, random_state=42)
```

After fixing:
```
Training Accuracy: 73%
Testing Accuracy:  56%
Gap:               17% — much better
```

### 4. The Bias-Variance Tradeoff

Finding the right `max_depth` by testing multiple values:

```
depth=  3 | train=0.62 | test=0.63 | gap=-0.004  ✅ no overfit but low recall
depth=  5 | train=0.62 | test=0.63 | gap=-0.004  ✅ good
depth= 10 | train=0.63 | test=0.63 | gap=-0.003  ✅ sweet spot
depth=150 | train=1.00 | test=0.53 | gap= 0.470  ❌ severe overfit
```

```
Too simple → underfitting → misses patterns → low accuracy
Sweet spot → just right  → learns patterns → best accuracy
Too complex → overfitting → memorizes data → fails on new data
```

This is called the **bias-variance tradeoff** — one of the most fundamental concepts in ML.

### 5. Feature Importances

Decision trees and random forests provide feature importances — a score showing how much each feature contributed to predictions. This is not available in logistic regression.

**Consistent across all models:**
```
shipping_cost_ngn:          35% — most important
quantity:                   27% — second most important
destination_city_enc:       12%
shipping_duration_expected: 11%
logistics_company_enc:       9%
origin_city_enc:             7%
```

Consistency across models confirms these features have **real signal** — not just noise.

### 6. Class Imbalance — Same Fix as Before

```python
dt_model = DecisionTreeClassifier(
    max_depth=10,
    class_weight='balanced',
    random_state=42
)
```

Without `class_weight='balanced'` the model predicted "on time" for everything and never detected delays — even with a good depth setting.

---

## Model Comparison — All Three Projects

| Model | Test Accuracy | Delayed Recall | On Time Recall | Overfitting Gap |
|---|---|---|---|---|
| Logistic Regression | 50% | 49% | 50% | — |
| Decision Tree | 56% | 25% | 75% | 17% |
| Random Forest | 52% | 42% | 57% | 9% |

### Which Model Wins?

| Priority | Best Model |
|---|---|
| Catching delays (recall) | Random Forest — 42% recall |
| Simplicity and explainability | Decision Tree |
| Speed in production | Logistic Regression |
| Reducing overfitting | Random Forest — smallest gap |
| Overall balance | Random Forest |

**For a shipment delay product — Random Forest wins** because it catches the most delays even though overall accuracy is lower. Missing a delay costs more than a false alarm.

### Why All Models Struggle

All three models perform around 50-56% — barely better than random guessing. This is not a model problem — it's a data problem. The dataset was synthetically generated with no real relationship between shipping features and delay status. No algorithm can find patterns that don't exist.

**Lesson:** Always validate that your data has real signal before spending time on model tuning.

---

## Tools & Libraries

```python
datasets                      # Hugging Face dataset loading
pandas                        # Data manipulation
numpy                         # Numerical operations
scikit-learn                  # Model training and evaluation
sklearn.tree.DecisionTreeClassifier
sklearn.ensemble.RandomForestClassifier
matplotlib                    # Visualisation
python-dotenv                 # Environment variable management
huggingface_hub               # Authentication
```

---

## How to Run

```bash
# Clone the repo
git clone https://github.com/shambhavichaugule/decision-tree-random-forest.git
cd decision-tree-random-forest

# Activate virtual environment
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt

# Add your Hugging Face token to .env
echo "HF_TOKEN=your_token_here" > .env

# Run decision tree model
python decision_tree.py
```

---

## PM Perspective

This project simulates a real product decision: **which model should we ship for delay prediction?**

**Model selection is a product decision, not just a technical one:**
- A model with 56% accuracy but 42% delay recall is more valuable than one with 63% accuracy and 0% delay recall
- Interpretability matters — operations teams need to understand why a shipment was flagged
- Overfitting means your demo looks great but production fails — always check the gap

**Feature importances as a product tool:**
- Shipping cost and quantity drive 62% of predictive power
- This tells operations teams where to focus interventions
- Low importance features (origin city at 7%) may not be worth the engineering cost to collect

**What I would do as a PM before shipping this:**
1. Validate data quality — synthetic labels with no real signal means the model is useless
2. Define the right metric upfront — accuracy vs recall is a business decision
3. Set a minimum recall threshold before launching — e.g. must catch at least 60% of delays
4. Monitor model performance in production — real world data will be different from training data

---

## Next Steps

- Find a dataset with real signal to see properly performing models
- Try XGBoost and LightGBM — industry standard gradient boosting models
- Explore hyperparameter tuning with GridSearchCV
---
