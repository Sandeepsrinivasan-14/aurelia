# Contributing

Thanks for your interest in Aurelia.

```bash
pip install -e ".[dev]"
make test      # pytest
make lint      # ruff
python -m eval.benchmark   # regenerate docs/benchmark.json
```

**Ground rules**

- Synthetic data only. Never commit real patient data, credentials or `.env` files.
- New retrieval strategies go in `src/aurelia/rag/strategies.py` and must be added to the benchmark with an honest result, including where they fail.
- Keep `ruff check .` and `pytest` green; CI enforces both.
- Aurelia is not a medical device. Don't present outputs as clinical advice.
