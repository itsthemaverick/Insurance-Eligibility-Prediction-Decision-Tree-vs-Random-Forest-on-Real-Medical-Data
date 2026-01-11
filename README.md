
# 🏥 Insurance Eligibility Prediction  
### Decision Tree vs Random Forest on Real Medical Data

A **reproducible Machine Learning project** that predicts **insurance eligibility** using the real-world **Medical Cost Personal Dataset**.  
The project compares an interpretable **Decision Tree** with a high-performance **Random Forest**, supported by exploratory data analysis, visualizations, and clean ML engineering practices.

---

## 📌 Problem Statement

Insurance providers must decide whether an applicant is **eligible for insurance coverage** based on demographic and health-related attributes.

Using real medical insurance data, this project:
- Reframes a **cost prediction problem** into a **binary eligibility classification task**
- Compares a simple interpretable model with a robust ensemble model
- Demonstrates the **bias–variance trade-off** in practice

---

## 📊 Dataset

**Medical Cost Personal Dataset** (Kaggle)

**Features**
- age: Age of the individual  
- sex: Gender  
- bmi: Body Mass Index  
- children: Number of dependents  
- smoker: Smoking status  
- region: Residential region  
- charges: Medical insurance cost  

### 🎯 Target Engineering

The original regression target (`charges`) is converted into a binary eligibility label:

eligible = 1 if charges < median(charges) else 0

This mirrors real-world insurance underwriting where higher expected cost implies higher risk.

---

## 🧠 Models Used

| Model | Description |
|-----|------------|
| Decision Tree | Interpretable baseline model |
| Random Forest | Ensemble model with reduced variance |

---

## 📁 Project Structure

```
insurance-eligibility-ml/
│
├── main.py
├── data/
│   └── insurance.csv
├── src/
│   ├── load_data.py
│   ├── preprocess.py
│   ├── decision_tree.py
│   ├── random_forest.py
│   └── evaluate.py
├── visualizations/
│   ├── eda.py
│   ├── model_performance.py
│   └── feature_importance.py
├── requirements.txt
└── README.md
```

---

## 📈 Model Performance

### Decision Tree
- Accuracy: **91.42%**
- Highly interpretable
- Slightly higher variance

### Random Forest
- Accuracy: **92.16%**
- Better generalization
- Reduced overfitting

| Metric | Decision Tree | Random Forest |
|------|---------------|---------------|
| Accuracy | 91.42% | **92.16%** |
| Interpretability | High | Medium |
| Variance | High | Low |

---

## 🧪 Classification Insight

- High recall for eligible applicants
- Balanced precision–recall trade-off
- Realistic performance without overfitting

---

## 🔁 Reproducibility

```bash
pip install -r requirements.txt
python main.py
```

- Fixed random seeds
- Modular pipeline
- Deterministic results

---

## 🧠 Key Learnings

- Decision Trees are powerful but unstable
- Random Forest reduces variance via ensembling
- Target engineering is crucial in applied ML
- Metrics matter more than raw accuracy
- Visualization improves trust and interpretability

---

## 🚀 Future Improvements

- Hyperparameter tuning (GridSearchCV)
- SHAP-based explainability
- Model deployment with FastAPI
- Comparison with Gradient Boosting models

---

## 🧑‍💻 Author

**Itsthemaverick**  
Machine Learning | Applied AI | Data Science

---

⭐ This project demonstrates **practical, business-aligned Machine Learning**, not just model training.
