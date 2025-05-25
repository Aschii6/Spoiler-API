FastAPI Server to process requests for predicting spoilers in text content.

### Body

```json
{
  "sentences": [
    "Spoiler sentence",
    "Sentence without spoilers"
  ],
  
  "titles": [
    "Title with spoilers",
    "Title without spoilers"
  ]
}
```

Titles is optional.

### Response

```json
{
  "predictions": [
    1,
    0
  ]
}
```
