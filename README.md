# CAPTCHA Solver

A Python mini-project to recognize text from SVG CAPTCHAs.

## Quick Start

```bash
# Install dependencies
pip install -r detect_captcha/requirements.txt

# Train the solver (interactive)
python -m detect_captcha.train

# Run the API server
uvicorn detect_captcha.main:app --reload
```

## API Usage

### Solve a CAPTCHA

```bash
# Using JSON
curl -X POST http://localhost:8000/solve \
  -H "Content-Type: application/json" \
  -d '{"svg": "<svg>...</svg>"}'

# Using raw SVG
curl -X POST http://localhost:8000/solve \
  -H "Content-Type: text/plain" \
  --data-binary @sample.svg
```

### Health Check

```bash
curl http://localhost:8000/
```

## How It Works

1. **Training**: Use `python -m detect_captcha.train` to fetch CAPTCHAs and label them manually.
2. **Matching**: The solver normalizes SVG paths to point clouds and finds the closest match in the database.
3. **API**: FastAPI serves the solution via a simple POST endpoint.

## Files

- `detect_captcha/main.py` - FastAPI server
- `detect_captcha/solver.py` - Recognition logic
- `detect_captcha/train.py` - Training CLI
- `detect_captcha/utils.py` - SVG parsing utilities
- `detect_captcha/database.json` - Knowledge base (grows as you train)
