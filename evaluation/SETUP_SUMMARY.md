# Setup Guide: Running IR Evaluation

## Prerequisites

- Python 3.8 or higher
- OpenAI API key

---

## Setup Steps

### 1. Create Virtual Environment

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set OpenAI API Key

**Linux/macOS:**
```bash
export OPENAI_API_KEY='your-api-key-here'
```

**Windows:**
```bash
set OPENAI_API_KEY=your-api-key-here
```

### 4. Run Evaluation

```bash
cd evaluation
python ir_evaluation_kg.py
```

---

## Output Files

The evaluation generates:
- `ir_evaluation_summary.csv` - Aggregate metrics
- `ir_evaluation_category_summary.csv` - Category-wise performance
- `ir_evaluation_results.json` - Detailed results
