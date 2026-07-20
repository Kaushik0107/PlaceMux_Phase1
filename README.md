# PlaceMux Phase 1

Machine Learning Environment Setup

Author: Kaushik B H

Industry Immersion Phase 1

## Task 3 – Adding a New Model

The project uses a modular architecture.

To replace the current baseline model:

1. Open `src/model/train.py`
2. Replace the `DummyClassifier` with another scikit-learn model.
3. Keep the remaining pipeline unchanged.
4. Run `python src/main.py`.

No changes are required in:
- `main.py`
- `evaluate.py`
- `config.py`

This allows new models to be integrated by modifying only one file.
