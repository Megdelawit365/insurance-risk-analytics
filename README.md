# End-to-End Insurance Risk Analytics & Predictive Modeling  

---

## Project Overview  
### Task 1  
### Task 2   
### Task 3  
### Task 4  

---

## Project Structure  
```bash
insurance-risk-analytics/
├── .github/
│   └── workflows/
│       └── ci.yml
├── data/                     # tracked by DVC, not Git
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_hypothesis_testing.ipynb
│   └── 03_modeling.ipynb
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── eda_utils.py
│   ├── hypothesis_tests.py
│   └── modeling.py
├── reports/
│   └── final_report.md
├── tests/
├── .dvc/
├── .gitignore
├── dvc.yaml
├── requirements.txt
└── README.md
```

---

## Setup Instrcutions
### 1. Clone the repository  
```bash
git clone https://github.com/Megdelawit365/insurance-risk-analytics
cd insurance-risk-analytics
```

### 2. Create and activate virtual environment
```bash
python venv venv
```
Windows:
```bash
venv\Scripts\activate
```
MAC/Linux:
```bash
source venv/bin/activat
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## Tools and Technologies Used