# End-to-End Insurance Risk Analytics & Predictive Modeling  

## Project Overview  

This project analyzes historical car insurance data from AlphaCare Insurance Solutions (ACIS) to uncover risk patterns, evaluate portfolio profitability, and support data-driven pricing decisions.

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


## Data Version Control (DVC)

This project uses DVC to manage large datasets separately from Git.

### Initialize DVC

```bash
dvc init
```

### Configure Local Storage

```bash
dvc remote add -d localstorage C:\dvc-storage
```

### Track Data

```bash
dvc add data/MachineLearningRating_v3.txt
```

### Push Data to Remote Storage

```bash
dvc push
```

### Reproduce Data

After cloning the repository:

```bash
dvc pull
```

## Tools and Technologies Used
pytest
flake8
dvc
numpy
pandas
matplotlib
seaborn