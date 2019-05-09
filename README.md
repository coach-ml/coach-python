# Coach Python SDK

## Initialization

```python
coach = new CoachClient(apiKey)
```

## Usage

### byte[] get(model)

This will download the model and store it as a byte array. This can be written to disk or kept in memory for future predictions.

```python
coach.get('flowers')
```

### string[] predict(model, image)

Accepts a model and image and returns a map of results.

```python
coach.predict(byte[], 'rose.jpg')
```

```python
coach.predict(byte[], byte[])
```