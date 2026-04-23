# Price Estimation Example

The model learns to return `$<price>` for a product description.

## Data format

Each line in the train/val/test JSONL files should be:

```json
{ "summary": "Compact wireless mouse with ergonomic grip ...", "price": 19.99 }
```

Populate `data/train.jsonl`, `data/validation.jsonl`, `data/test.jsonl` before running.

If you prefer to load from a Hugging Face dataset, edit `config.yaml`:

```yaml
data:
  source: hf
  path: arneesh/items_lite
  train_split: train
  val_split: validation
  test_split: test
  max_train: 100
  max_val: 50
  max_test: 200
```

(Install the optional dependency first: `uv sync --extra huggingface`.)

## Running

### CLI

```bash
openai-ft run examples/price_estimation/config.yaml \
    --formatter examples.price_estimation.formatter:PriceFormatter \
    --metric   examples.price_estimation.formatter:price_regression_metric
```

### Python

```bash
python -m examples.price_estimation.run
```

The pipeline writes JSONL files, uploads them, launches a fine-tuning job,
waits for it to finish, evaluates the resulting model on the test split, and
saves everything under `artifacts/price-estimation/`.
