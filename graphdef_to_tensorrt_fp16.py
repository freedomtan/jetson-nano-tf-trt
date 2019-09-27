from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import numpy as np
import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
import time

def convert_graphdef(model_file, output_layer, output_file):
  graph = tf.Graph()
  graph_def = tf.compat.v1.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())

  with graph.as_default():
    tf.import_graph_def(graph_def)
    trt_graph = trt.create_inference_graph(input_graph_def = graph_def, 
      outputs=[output_layer], precision_mode='FP16')

    tf.io.write_graph(trt_graph, '/tmp/', output_file, as_text=False)

  return trt_graph

if __name__ == "__main__":
  model_file = "/tmp/mobilenet_v1_1.0_224_frozen.pb"
  input_layer = "input"
  output_layer = "MobilenetV1/Predictions/Reshape_1"
  output_file = "trt_graph_fp16.pb"

  parser = argparse.ArgumentParser()
  parser.add_argument("--graph", help="graph/model to be executed")
  parser.add_argument("--input_layer", help="name of input layer")
  parser.add_argument("--output_layer", help="name of output layer")
  parser.add_argument("--output_file", help="name of output file")
  args = parser.parse_args()

  if args.graph:
    model_file = args.graph
  if args.input_layer:
    input_layer = args.input_layer
  if args.output_layer:
    output_layer = args.output_layer
  if args.output_file:
    output_file = args.output_file

  convert_graphdef(model_file, output_layer, output_file)
