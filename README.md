# Coach Python SDK

Coach is an end-to-end Image Recognition platform, we provide the tooling to do effective data collection, training, and on-device parsing of Image Recognition models.

The Python 3 SDK interacts with the Coach web API in order to download and parse your trained Coach models.

## Installing
Install and update using [pip](https://pip.pypa.io/en/stable/quickstart/):
```bash
pip install -U coach
```

## Usage

Coach can be initialized 2 different ways. If you are only using the offline model parsing capabilities and already have a model package on disk, you can initialize like so:

```python
coach = Coach()

# We already had the `flowers` model on disk, no need to authenticate:
result = coach.get_model('flowers').predict('rose.jpg')
```

However, in order to download your trained models, you must authenticate with your API key:
```python
coach = Coach().login('myapikey')

# Now that we're authenticated, we can cache our models for future use:
coach.cache_model('flowers')

# Evaluate with our cached model:
result = coach.get_model('flowers').predict('rose.jpg')
```

## API Breakdown

### Coach
`__init__(self, is_debug=False)`  
Optional `is_debug`, if `True`, additional logs will be displayed

`login(self, apiKey) -> Coach`  
Authenticates with Coach service and allows for model caching. Accepts API Key as its only parameter. Returns its own instance.

`cache_model(self, name, path='.')`  
Downloads model from Coach service to disk. Specify the name of the model, and the path to store it. This will create a new directory in the specified path and store any model related documents there.

`get_model(self, path) -> CoachModel`
Loads model into memory. Specify the path of the cached models directory. Returns a `CoachModel`

### CoachModel
`__init__(self, graph, labels, base_module)`  
Initializes a new instance of `CoachModel`, accepts a loaded `tf.Graph()`, array of `labels`, and the `base_module` the graph was trained off of.

`predict(self, image) -> dict`  
Specify the directory of an image file or the image as a byte array. Parses the specified image into memory and runs it through the loaded model. Returns a dict of its predictions in order of confidence.