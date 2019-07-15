#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os
import csv
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

    def __read_tensor_from_bytes(self, imageBytes):
        image_reader = tf.image.decode_image(imageBytes, channels=3)
        float_caster = tf.cast(image_reader, tf.float32)

        dims_expander = tf.expand_dims(float_caster, 0)
        resized = tf.image.resize_bilinear(dims_expander, [self.input_height, self.input_width])
        #normalized = tf.divide(tf.subtract(resized, [0]), [299])
        normalized = resized
        sess = tf.Session()
        result = sess.run(normalized)

        return result

    def __read_tensor_from_image_file(self, file_name):
        tensor = tf.read_file(file_name, name="file_reader")
        return self.__read_tensor_from_bytes(tensor)        

    def predict(self, image):
        input_name = "input"
        output_name = "output"

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

        js = {};
        top_k = results.argsort()[-5:][::-1]
        for i in top_k:
            js[self.labels[i]] = results[i]

        return js

class Coach:
    def __init__(self, is_debug=False):
        self.is_debug = is_debug

    def login(self, apiKey):
        self.apiKey = apiKey
        self.id = apiKey[0:5]
        self.profile = self.__get_profile()
        self.bucket = self.profile['bucket']
        return self

    def __is_authenticated(self):
        return self.apiKey != None and self.id != None and self.bucket != None

    def __get_profile(self):
        url = 'https://2hhn1oxz51.execute-api.us-east-1.amazonaws.com/prod/' + self.id
        response = requests.get(url, headers={"X-Api-Key": self.apiKey}).json()
        return response

    # Downloads model
    def cache_model(self, name, path='.'):
        if not self.__is_authenticated():
            print('You must login to cache a model')
            return

        # Create dir to store model files
        try:
            # Create target Directory
            os.mkdir(name)
        except FileExistsError:
            pass

        models = self.profile['models']

        model = ''
        for _model in models:
            if _model['name'] == name:
                model = _model

        #model = models[models.index(name)]
        print(model)

        profile_version = model['version']
        profile_path = path + '/' + name + '/manifest.json'
        if os.path.isfile(profile_path):
            _p = open(profile_path, 'r')
            local_profile = json.loads(_p.read())
            _p.close()

            if local_profile['version'] == profile_version:
                if self.is_debug:
                    print('Version match, skipping download')
                return
        else:
            _p = open(profile_path, 'w')
            _p.write(json.dumps(model))
            _p.close()

        url = 'https://la41byvnkj.execute-api.us-east-1.amazonaws.com/prod/' + self.bucket + '/model-bin?object=trained/' + name + '/' + str(profile_version) + '/model'

        m_file = 'frozen.pb'
        m_url = url + '/' + m_file
        # Write bin to path
        m_response = requests.get(m_url, headers={"X-Api-Key": self.apiKey, "Accept": "", "Content-Type": "application/octet-stream"}).content

        model_path = path + '/' + name + '/' + m_file
        model = open(model_path, 'wb')
        model.write(m_response)
        model.close()

    def get_model(self, path):
        graph = tf.Graph()
        graph_def = tf.GraphDef()
        with open(path + '/frozen.pb', "rb") as f:
            #text_format.Merge(f.read(), graph_def)
            graph_def.ParseFromString(f.read())
        with graph.as_default():
            tf.import_graph_def(graph_def)

        manifest_path = path + '/manifest.json'
        m = open(manifest_path, 'r')
        manifest = json.loads(m.read())
        
        # Load lables
        labels = manifest['labels']
        base_module = manifest['module']

        return CoachModel(graph, labels, base_module)

    def get_model_remote(self, name, path="."):
        cache_model(name, path)
        return get_model(path)