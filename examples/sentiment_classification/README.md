# Sentiment Classification Example

Shows how the same framework fine-tunes a model for a totally different task:
classifying text as positive / negative / neutral.

## Data format

Each line in the JSONL files:

```json
{"text": "I love this product!", "label": "positive"}
```

## Running

```bash
openai-ft run examples/sentiment_classification/config.yaml \
    --formatter examples.sentiment_classification.formatter:SentimentFormatter \
    --metric   examples.sentiment_classification.formatter:sentiment_accuracy
```

or

```bash
python -m examples.sentiment_classification.run
```
