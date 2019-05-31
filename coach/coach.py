#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os
import csv
import numpy as np
import requests

class CoachModel:
    def __init__(graph, labels):
        self.graph = graph
        self.lables = labels

    def __read_tensor_from_image_file(file_name, input_height=224, input_width=224, input_mean=0, input_std=255):
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

    def predict(image):
        output_name = "import/softmax_input/Softmax"
        input_name = "import/lambda_input_input"

        input_operation = self.graph.get_operation_by_name(input_name)
        output_operation = self.graph.get_operation_by_name(output_name)

        t = __read_tensor_from_image_file(image)
        
        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            results = sess.run(output_operation.outputs[0], {
                input_operation.outputs[0]: t
            })
        results = np.squeeze(results)

        # TODO: Return a JSON object
        top_k = results.argsort()[-5:][::-1]
        for i in top_k:
            print(labels[i], results[i])

class Coach:

    def __init__(apiKey):
        id = apiKey[0:5]
        url = f'https://2hhn1oxz51.execute-api.us-east-1.amazonaws.com/prod/{id}'
        response = requests.get(url, headers={"X-Api-Key": apiKey}).json()
        print(response)
        self.bucket = '' # Use bucket from response

    # Downloads model
    def cache_model(name, version, path='.'):
        base = f'https://la41byvnkj.execute-api.us-east-1.amazonaws.com/prod/{self.bucket}/model-bin?object=trained/{name}/{version}/frozen.pb'
        # Write bin to path

    # Downloads and loads model into memory
    def get_model_remote(name, version, path='.'):
        cache_model(name, version, path)
        return get_model(path)

    def get_model(path):
        graph = tf.Graph()
        graph_def = tf.GraphDef()
        with open(path, "rb") as f:
            #text_format.Merge(f.read(), graph_def)
            graph_def.ParseFromString(f.read())
        with graph.as_default():
            tf.import_graph_def(graph_def)

        # Load lables

        return CoachModel(graph, [])

    def load_labels(label_file):
        label = []
        proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
        for l in proto_as_ascii_lines:
            label.append(l.rstrip())
        return label