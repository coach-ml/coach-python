#!/usr/bin/env python3

import tensorflow as tf
import os
import numpy as np
import requests
import json

class CoachModel:
    def __init__(self, graph, labels, base_module):
        self.graph = graph
        self.labels = labels

        mod_len = len(base_module)

        input_size = int(base_module[mod_len-3:mod_len])
        self.input_height = input_size
        self.input_width = input_size
        #self.input_mean = 0
        #self.input_std = 299

        # handle expected module sizes
        if (base_module == ""):
            pass

    def __read_tensor_from_bytes(self, image_bytes):
        image_reader = tf.image.decode_image(image_bytes, channels=3)
        float_caster = tf.cast(image_reader, tf.float32)

        dims_expander = tf.expand_dims(float_caster, 0)
        resized = tf.image.resize_bilinear(dims_expander, [self.input_height, self.input_width])
        #normalized = tf.divide(tf.subtract(resized, [self.input_mean]), [self.input_std])
        normalized = resized
        sess = tf.Session()
        result = sess.run(normalized)

        return result

    def __read_tensor_from_image_file(self, file_name):
        tensor = tf.read_file(file_name, name="file_reader")
        return self.__read_tensor_from_bytes(tensor)        

    def predict(self, image, input_name="input", output_name="output"):
        if not os.path.isfile(image):
            raise ValueError(f'Invalid image: {image}')

        input_operation = self.graph.get_operation_by_name(input_name)
        output_operation = self.graph.get_operation_by_name(output_name)

        if type(image) is str:
            t = self.__read_tensor_from_image_file(image)
        else:
            t = self.__read_tensor_from_bytes(image)
        
        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            results = sess.run(output_operation.outputs[0], {
                input_operation.outputs[0]: t
            })
        results = np.squeeze(results)

        js = {}
        top_k = results.argsort()[-5:][::-1]
        for i in top_k:
            js[self.labels[i]] = results[i]

        return js

class CoachClient:
    def __init__(self, is_debug=False):
        self.is_debug = is_debug

    def login(self, apiKey):
        self.apiKey = apiKey
        self.id = apiKey[0:5]
        try:
            self.profile = self.__get_profile()
        except Exception:
            raise ValueError("Failed to login, check your API key")
        
        self.bucket = self.profile['bucket']
        return self

    def __is_authenticated(self):
        return self.apiKey != None and self.id != None and self.bucket != None

    def __get_profile(self):
        url = 'https://2hhn1oxz51.execute-api.us-east-1.amazonaws.com/prod/' + self.id
        response = requests.get(url, headers={"X-Api-Key": self.apiKey})
        response.raise_for_status()     
        return response.json()

    # Downloads model
    def cache_model(self, model_name, path='.', skip_match=True, model_type='frozen'):
        if not self.__is_authenticated():
            raise ValueError('You must login to cache a model')
        if not os.path.isdir(path):
            raise ValueError(f'{path} is not a valid directory')

        models = self.profile['models']

        model = ''
        for _model in models:
            if _model['name'] == model_name:
                model = _model

        # TODO: Better versioning with labels
        version = 0
        if version <= 0:
            version = model['version']
        else:
            model['version'] = version

        model_dir = os.path.join(path, model_name)
        profile_path = os.path.join(model_dir, 'manifest.json')
        
        if os.path.isfile(profile_path):
            _p = open(profile_path, 'r')
            local_profile = json.loads(_p.read())
            _p.close()

            if local_profile['version'] == version and skip_match:
                if self.is_debug:
                    print('Version match, skipping download')
                return
        
        elif not os.path.isdir(model_dir):
            os.mkdir(model_dir)

        _p = open(profile_path, 'w')
        _p.write(json.dumps(model))
        _p.close()

        url = f'https://la41byvnkj.execute-api.us-east-1.amazonaws.com/prod/{self.bucket}'

        model_filename = None
        if model_type == 'frozen':
            model_filename = 'frozen.pb'
        elif model_type == 'unity':
            model_filename = 'unity.bytes'
        elif model_type == 'mobile':
            model_filename = 'mobile.tflite'
        else:
            raise ValueError(f'model_type {model_type} is invalid. Can be one of: frozen, unity, mobile')

        # Write bin to path
        try:
            m_response = requests.get(url, params={"object": f"trained/{model_name}/{str(version)}/model/{model_filename}", }, headers={"X-Api-Key": self.apiKey, "Accept": "", "Content-Type": "application/octet-stream"})
            m_response.raise_for_status()
            
            content = m_response.content
        except Exception:
            raise ValueError(f'Failed to cache model: {model_name}')

        model_path = os.path.join(model_dir, model_filename)
        model = open(model_path, 'wb')
        model.write(content)
        model.close()

    def get_model(self, path):
        if not os.path.isdir(path):
            raise ValueError(f'Invalid model directory: {path}')

        model_type = 'frozen.pb'
        graph = tf.Graph()
        graph_def = tf.GraphDef()
        with open(os.path.join(path, model_type), "rb") as f:
            graph_def.ParseFromString(f.read())
        with graph.as_default():
            tf.import_graph_def(graph_def)

        manifest_path = os.path.join(path, 'manifest.json')
        m = open(manifest_path, 'r')
        manifest = json.loads(m.read())
        
        # Load lables
        labels = manifest['labels']
        base_module = manifest['module']

        return CoachModel(graph, labels, base_module)

    def get_model_remote(self, model_name, path="."):
        self.cache_model(model_name, path)
        return self.get_model(path)
