#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os
import csv
import numpy as np
import requests
import json

class CoachModel:
    def __init__(self, graph, labels):
        self.graph = graph
        self.labels = labels

    def __read_tensor_from_image_file(self, file_name, input_height=224, input_width=224, input_mean=0, input_std=255):
        input_name = "file_reader"
        output_name = "normalized"
        file_reader = tf.read_file(file_name, input_name)
        if file_name.endswith(".png"):
            image_reader = tf.image.decode_png(
                file_reader, channels=3, name="png_reader")
        elif file_name.endswith(".gif"):
            image_reader = tf.squeeze(
                tf.image.decode_gif(file_reader, name="gif_reader"))
        elif file_name.endswith(".bmp"):
            image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
        else:
            image_reader = tf.image.decode_jpeg(
                file_reader, channels=3, name="jpeg_reader")
        float_caster = tf.cast(image_reader, tf.float32)
        dims_expander = tf.expand_dims(float_caster, 0)
        resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
        normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
        sess = tf.Session()
        result = sess.run(normalized)

        return result

    def predict(self, image):
        output_name = "import/softmax_input/Softmax"
        input_name = "import/lambda_input_input"

        input_operation = self.graph.get_operation_by_name(input_name)
        output_operation = self.graph.get_operation_by_name(output_name)

        t = self.__read_tensor_from_image_file(image)
        
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

    def login(self, apiKey):
        self.apiKey = apiKey
        self.id = apiKey[0:5]
        profile = self.__get_profile()
        self.bucket = profile['bucket']
        return self

    def __is_authenticated(self):
        return self.apiKey != None and self.id != None and self.bucket != None

    def __get_profile(self):
        url = f'https://2hhn1oxz51.execute-api.us-east-1.amazonaws.com/prod/{self.id}'
        response = requests.get(url, headers={"X-Api-Key": self.apiKey}).json()
        return response

    # Downloads model
    def cache_model(self, name, version, path='.'):
        if not self.__is_authenticated():
            print('You must login to cache a model')
            return

        # Create dir to store model files
        try:
            # Create target Directory
            os.mkdir(name)
        except FileExistsError:
            pass

        profile_path = f'{path}/{name}/manifest.json'
        if os.path.isfile(profile_path):
            _p = open(profile_path, 'r')
            profile = json.loads(_p.read())
            _p.close()

            if version == profile[name]['version']:
                print('Version match, skipping download')
                return
        else:
            profile = self.__get_profile()
            p_to_write = profile['models'][name]
            p_to_write = { f'{name}': p_to_write }

            _p = open(profile_path, 'w')
            _p.write(json.dumps(p_to_write))
            _p.close()

        url = f'https://la41byvnkj.execute-api.us-east-1.amazonaws.com/prod/{self.bucket}/model-bin?object=trained/{name}/{version}/model'

        m_file = 'frozen.pb'
        m_url = f'{url}/{m_file}'
        # Write bin to path
        m_response = requests.get(m_url, headers={"X-Api-Key": self.apiKey, "Accept": "", "Content-Type": "application/octet-stream"}).content

        model_path = f'{path}/{name}/{m_file}'
        model = open(model_path, 'wb')
        model.write(m_response)
        model.close()

        # Write label to path
        l_file = 'labels.csv'
        l_url = f'{url}/{l_file}'
        # Write label to path
        l_response = requests.get(l_url, headers={"X-Api-Key": self.apiKey}).text

        l_path = f'{path}/{name}/{l_file}'
        label = open(l_path, 'w')
        label.write(l_response)
        label.close()

    # Downloads and loads model into memory
    def get_model_remote(self, name, version, path='.'):
        if not self.__is_authenticated():
            print('You must login to cache a model')
            return

        self.cache_model(name, version, path)
        return get_model(path)

    def get_model(self, path):
        graph = tf.Graph()
        graph_def = tf.GraphDef()
        with open(f'{path}/frozen.pb', "rb") as f:
            #text_format.Merge(f.read(), graph_def)
            graph_def.ParseFromString(f.read())
        with graph.as_default():
            tf.import_graph_def(graph_def)

        # Load lables
        labels = self.__load_labels(f'{path}/labels.csv')

        return CoachModel(graph, labels)

    def __load_labels(self, label_file):
        label = []
        proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
        for l in proto_as_ascii_lines:
            label.append(l.rstrip())
        return label
