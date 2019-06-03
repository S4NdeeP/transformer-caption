# coding=utf-8
# Copyright 2018 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MS COCO."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import json
import os
import random
import zipfile

# Dependency imports

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import image_utils
from tensor2tensor.data_generators import imagenet
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.utils import registry
from tensor2tensor.utils import metrics

import tensorflow as tf


# FOr botom-up
import base64
import numpy as np
import csv
import sys
import zlib
import time
import mmap

use_bottom_up_features = True

csv.field_size_limit(sys.maxsize)

# URLs and filenames for MSCOCO data.
_MSCOCO_ROOT_URL = "http://msvocds.blob.core.windows.net/"
_MSCOCO_URLS = [
    "coco2014/train2014.zip", "coco2014/val2014.zip", "coco2014/test2014.zip",
    "annotations-1-0-3/captions_train-val2014.zip"
]
_MSCOCO_TRAIN_PREFIX = "train2014"
_MSCOCO_EVAL_PREFIX = "val2014"
#_MSCOCO_TEST_SERV_PREFIX = "test_serv2014"
_MSCOCO_TEST_SERV_PREFIX = "val2014"

_MSCOCO_TRAIN_CAPTION_FILE = "annotations/captions_train2014.json"
_MSCOCO_EVAL_CAPTION_FILE = "annotations/captions_val2014.json"
#_MSCOCO_EVAL_SERVER_IMAGE_INFO_FILE = "annotations/image_info_test2014.json"
_MSCOCO_EVAL_SERVER_IMAGE_INFO_FILE = "annotations/captions_val2014_serv.json"

_MSCOCO_TRAIN_BOTTOM_UP_FEATURE_FILE = ["bottom_up_features/trainval/karpathy_train_resnet101_faster_rcnn_genome.tsv.0","bottom_up_features/trainval/karpathy_train_resnet101_faster_rcnn_genome.tsv.1"]
#_MSCOCO_TRAIN_BOTTOM_UP_FEATURE_FILE = ["bottom_up_features/trainval/karpathy_train_resnet101_faster_rcnn_genome.tsv.1.4"]
print(_MSCOCO_TRAIN_BOTTOM_UP_FEATURE_FILE)

_MSCOCO_EVAL_BOTTOM_UP_FEATURE_FILE = ["bottom_up_features/trainval/karpathy_val_resnet101_faster_rcnn_genome.tsv"]
#_MSCOCO_TEST_SERV_BOTTOM_UP_FEATURE_FILE = ["bottom_up_features/test/test2014/test2014_resnet101_faster_rcnn_genome.tsv.2"]
_MSCOCO_TEST_SERV_BOTTOM_UP_FEATURE_FILE = ["bottom_up_features/trainval/karpathy_test_resnet101_faster_rcnn_genome.tsv"]

def _get_mscoco(directory):
  """Download and extract MSCOCO datasets to directory unless it is there."""
  print("Setup MSCOCO data manually in data/tmp folder")
  for url in _MSCOCO_URLS:
    filename = os.path.basename(url)
    download_url = os.path.join(_MSCOCO_ROOT_URL, url)
    path = generator_utils.maybe_download(directory, filename, download_url)
    unzip_dir = os.path.join(directory, filename.strip(".zip"))
    if not tf.gfile.Exists(unzip_dir):
      zipfile.ZipFile(path, "r").extractall(directory)

def mscoco_generator_test_server(data_dir,
                     tmp_dir,
                     training,
                     how_many,
                     start_from=0,
                     eos_list=None,
                     vocab_filename=None):
  """Image generator for MSCOCO captioning problem with token-wise captions.

  Args:
    data_dir: path to the data directory.
    tmp_dir: path to temporary storage directory.
    training: a Boolean; if true, we use the train set, otherwise the test set.
    how_many: how many images and labels to generate.
    start_from: from which image to start.
    eos_list: optional list of end of sentence tokens, otherwise use default
      value `1`.
    vocab_filename: file within `tmp_dir` to read vocabulary from.

  Yields:
    A dictionary representing the images with the following fields:
    * image/encoded: the string encoding the image as JPEG,
    * image/format: the string "jpeg" representing image format,
    * image/class/label: a list of integers representing the caption,
    * image/height: an integer representing the height,
    * image/width: an integer representing the width.
    Every field is actually a list of the corresponding type.
  """
  if training:
    eos_list = [1] if eos_list is None else eos_list

    def get_vocab():
      """Get vocab for caption text encoder."""
      if data_dir is not None and vocab_filename is not None:
        vocab_filepath = os.path.join(data_dir, vocab_filename)
        if tf.gfile.Exists(vocab_filepath):
          tf.logging.info("Found vocab file: %s", vocab_filepath)
          vocab_symbolizer = text_encoder.SubwordTextEncoder(vocab_filepath)
          return vocab_symbolizer
        else:
          raise ValueError("Vocab file does not exist: %s", vocab_filepath)
      return None

    vocab_symbolizer = get_vocab()
    _get_mscoco(tmp_dir)
    caption_filepath = (
      _MSCOCO_TRAIN_CAPTION_FILE if training else _MSCOCO_EVAL_CAPTION_FILE)
    caption_filepath = os.path.join(tmp_dir, caption_filepath)
    prefix = _MSCOCO_TRAIN_PREFIX if training else _MSCOCO_EVAL_PREFIX
    caption_file = io.open(caption_filepath)
    caption_json = json.load(caption_file)
    # Dictionary from image_id to ((filename, height, width), captions).
    image_dict = dict()
    for image in caption_json["images"]:
      image_dict[image["id"]] = [(image["file_name"], image["height"],
                                  image["width"]), [], []]
    annotations = caption_json["annotations"]
    annotation_count = len(annotations)
    image_count = len(image_dict)
    tf.logging.info("Processing %d images and %d labels\n" % (image_count,
                                                              annotation_count))
    for annotation in annotations:
      image_id = annotation["image_id"]
      image_dict[image_id][1].append(annotation["caption"])

    # Reading Bottom-Up features
    FIELDNAMES = ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features']
    bottom_up_feature_filepaths = (
      _MSCOCO_TRAIN_BOTTOM_UP_FEATURE_FILE if training else _MSCOCO_EVAL_BOTTOM_UP_FEATURE_FILE)
    bottom_up_feature_filepaths = [os.path.join(tmp_dir, bottom_up_feature_filepath) for bottom_up_feature_filepath in
                                bottom_up_feature_filepaths]

    for bottom_up_feature_filepath in bottom_up_feature_filepaths:
      with open(bottom_up_feature_filepath, "r+b") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=FIELDNAMES)
        for item in reader:
          item['image_id'] = int(item['image_id'])
          item['image_h'] = int(item['image_h'])
          item['image_w'] = int(item['image_w'])
          item['num_boxes'] = int(item['num_boxes'])
          for field in ['boxes', 'features']:
            item[field] = np.frombuffer(base64.decodestring(item[field]),
                                        dtype=np.float32).tolist()
          image_dict[item['image_id']][2] = item

    data = list(image_dict.values())[start_from:start_from + how_many]
    random.shuffle(data)
    image_id = 0

    for image_info, labels, bottom_up_data in data:
      if not bottom_up_data:
        continue
      image_filename = image_info[0]
      image_filepath = os.path.join(tmp_dir, prefix, image_filename)
      with tf.gfile.Open(image_filepath, "r") as f:
        encoded_image_data = f.read()
        height, width = image_info[1], image_info[2]
        if training:
          for label in labels:
            label = label.replace('.', '').replace(',', '').replace("'", "").replace('"', '').replace('!', '')
            label = label.replace('&', 'and').replace('(', '').replace(")", "").replace('-', ' ').replace(';', '')
            label = label.replace('#', '').replace(':', '').replace('?', '')
            label = " ".join(label.split())  # replace multiple spaces
            label = label.lower()
            if vocab_filename is None or vocab_symbolizer is None:
              label = [ord(c) for c in label] + eos_list
            else:
              label = vocab_symbolizer.encode(label) + eos_list

            yield {
              "image/encoded": [encoded_image_data],
              "image/features": bottom_up_data['features'],
              "image/boxes": bottom_up_data['boxes'],
              "image/num_boxes": [bottom_up_data['num_boxes']],
              "image/format": ["jpeg"],
              "image/class/label": label,
              "image/height": [height],
              "image/width": [width]
            }
        else:
          tokenised_labels = []
          processed_labels = []
          for label in labels:
            label = label.replace('.', '').replace(',', '').replace("'", "").replace('"', '').replace('!', '')
            label = label.replace('&', 'and').replace('(', '').replace(")", "").replace('-', ' ').replace(';', '')
            label = label.replace('#', '').replace(':', '').replace('?', '')
            label = " ".join(label.split())  # replace multiple spaces
            label = label.lower()
            label_utf = label.encode("utf8")
            processed_labels.append(label_utf)
            if vocab_filename is None or vocab_symbolizer is None:
              label = [ord(c) for c in label] + eos_list
            else:
              # print(image_filepath + ':' + label)
              label = vocab_symbolizer.encode(label) + eos_list
            tokenised_labels.append(label)
          image_id += 1
          val_caption_path = os.path.join(data_dir, "val_caption.txt")
          with open(val_caption_path, "a") as myfile:
            myfile.write(str(image_id) + ':' + str(processed_labels) + '\n')

          val_image_path = os.path.join(data_dir, "val_image_id.txt")
          with open(val_image_path, "a") as myfile:
            myfile.write(str(bottom_up_data['image_id']) + '\n')
          yield {
            "image/encoded": [encoded_image_data],
            "image/features": bottom_up_data['features'],
            "image/boxes": bottom_up_data['boxes'],
            "image/num_boxes": [bottom_up_data['num_boxes']],
            "image/format": ["jpeg"],
            "image/class/label": tokenised_labels[0],
            "image/class/label_image": [image_id],
            "image/height": [height],
            "image/width": [width]
          }
  else:
    eos_list = [1] if eos_list is None else eos_list

    def get_vocab():
      """Get vocab for caption text encoder."""
      if data_dir is not None and vocab_filename is not None:
        vocab_filepath = os.path.join(data_dir, vocab_filename)
        if tf.gfile.Exists(vocab_filepath):
          tf.logging.info("Found vocab file: %s", vocab_filepath)
          vocab_symbolizer = text_encoder.SubwordTextEncoder(vocab_filepath)
          return vocab_symbolizer
        else:
          raise ValueError("Vocab file does not exist: %s", vocab_filepath)
      return None

    vocab_symbolizer = get_vocab()
    _get_mscoco(tmp_dir)
    caption_filepath = _MSCOCO_EVAL_SERVER_IMAGE_INFO_FILE
    caption_filepath = os.path.join(tmp_dir, caption_filepath)
    prefix = _MSCOCO_TEST_SERV_PREFIX
    caption_file = io.open(caption_filepath)
    caption_json = json.load(caption_file)
    # Dictionary from image_id to ((filename, height, width), captions).
    image_dict = dict()
    for image in caption_json["images"]:
      image_dict[image["id"]] = [(image["file_name"], image["height"],
                                  image["width"]),  []]

    image_count = len(image_dict)
    tf.logging.info("Processing %d images\n" % (image_count))

    # Reading Bottom-Up features
    FIELDNAMES = ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features']
    bottom_up_feature_filepaths = _MSCOCO_TEST_SERV_BOTTOM_UP_FEATURE_FILE
    bottom_up_feature_filepaths = [os.path.join(tmp_dir, bottom_up_feature_filepath) for bottom_up_feature_filepath in
                                bottom_up_feature_filepaths]

    for bottom_up_feature_filepath in bottom_up_feature_filepaths:
      with open(bottom_up_feature_filepath, "r+b") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=FIELDNAMES)
        for item in reader:
          try:
            item['image_id'] = int(item['image_id'])
            item['image_h'] = int(item['image_h'])
            item['image_w'] = int(item['image_w'])
            item['num_boxes'] = int(item['num_boxes'])
            for field in ['boxes', 'features']:
              item[field] = np.frombuffer(base64.decodestring(item[field]),
                                          dtype=np.float32).tolist()
            image_dict[item['image_id']][1] = item
          except Exception as e:
            print('Oops!'+ str(item['image_id']))

    data = list(image_dict.values())[start_from:start_from + how_many]

    for image_info, bottom_up_data in data:
      if not bottom_up_data:
        continue
      image_filename = image_info[0]
      image_filepath = os.path.join(tmp_dir, prefix, image_filename)
      with tf.gfile.Open(image_filepath, "r") as f:
        encoded_image_data = f.read()
        height, width = image_info[1], image_info[2]

        yield {
          "image/encoded": [encoded_image_data],
          "image/features": bottom_up_data['features'],
          "image/boxes": bottom_up_data['boxes'],
          "image/num_boxes": [bottom_up_data['num_boxes']],
          "image/format": ["jpeg"],
          "image/class/label": [0],
          "image/class/label_image": [int(bottom_up_data['image_id'])],
          "image/height": [height],
          "image/width": [width]
        }

def mscoco_generator(data_dir,
                     tmp_dir,
                     training,
                     how_many,
                     start_from=0,
                     eos_list=None,
                     vocab_filename=None):
  """Image generator for MSCOCO captioning problem with token-wise captions.

  Args:
    data_dir: path to the data directory.
    tmp_dir: path to temporary storage directory.
    training: a Boolean; if true, we use the train set, otherwise the test set.
    how_many: how many images and labels to generate.
    start_from: from which image to start.
    eos_list: optional list of end of sentence tokens, otherwise use default
      value `1`.
    vocab_filename: file within `tmp_dir` to read vocabulary from.

  Yields:
    A dictionary representing the images with the following fields:
    * image/encoded: the string encoding the image as JPEG,
    * image/format: the string "jpeg" representing image format,
    * image/class/label: a list of integers representing the caption,
    * image/height: an integer representing the height,
    * image/width: an integer representing the width.
    Every field is actually a list of the corresponding type.
  """
  eos_list = [1] if eos_list is None else eos_list
  def get_vocab():
    """Get vocab for caption text encoder."""
    if data_dir is not None and vocab_filename is not None:
      vocab_filepath = os.path.join(data_dir, vocab_filename)
      if tf.gfile.Exists(vocab_filepath):
        tf.logging.info("Found vocab file: %s", vocab_filepath)
        vocab_symbolizer = text_encoder.SubwordTextEncoder(vocab_filepath)
        return vocab_symbolizer
      else:
        raise ValueError("Vocab file does not exist: %s", vocab_filepath)
    return None

  vocab_symbolizer = get_vocab()
  _get_mscoco(tmp_dir)
  caption_filepath = (
      _MSCOCO_TRAIN_CAPTION_FILE if training else _MSCOCO_EVAL_CAPTION_FILE)
  caption_filepath = os.path.join(tmp_dir, caption_filepath)
  prefix = _MSCOCO_TRAIN_PREFIX if training else _MSCOCO_EVAL_PREFIX
  caption_file = io.open(caption_filepath)
  caption_json = json.load(caption_file)
  # Dictionary from image_id to ((filename, height, width), captions).
  image_dict = dict()
  for image in caption_json["images"]:
    image_dict[image["id"]] = [(image["file_name"], image["height"],
                                image["width"]), [], []]
  annotations = caption_json["annotations"]
  annotation_count = len(annotations)
  image_count = len(image_dict)
  tf.logging.info("Processing %d images and %d labels\n" % (image_count,
                                                            annotation_count))
  for annotation in annotations:
    image_id = annotation["image_id"]
    image_dict[image_id][1].append(annotation["caption"])

  # Reading Bottom-Up features
  FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features']
  bottom_up_feature_filepaths = (
      _MSCOCO_TRAIN_BOTTOM_UP_FEATURE_FILE if training else _MSCOCO_EVAL_BOTTOM_UP_FEATURE_FILE)
  bottom_up_feature_filepaths = [os.path.join(tmp_dir, bottom_up_feature_filepath) for bottom_up_feature_filepath in bottom_up_feature_filepaths]

  for bottom_up_feature_filepath in bottom_up_feature_filepaths:
    with open(bottom_up_feature_filepath, "r+b") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = FIELDNAMES)
        for item in reader:
            item['image_id'] = int(item['image_id'])
            item['image_h'] = int(item['image_h'])
            item['image_w'] = int(item['image_w'])
            item['num_boxes'] = int(item['num_boxes'])
            for field in ['boxes', 'features']:
                item[field] = np.frombuffer(base64.decodestring(item[field]),
                      dtype=np.float32).tolist()
            image_dict[item['image_id']][2] = item

  data = list(image_dict.values())[start_from:start_from + how_many]
  random.shuffle(data)
  image_id = 0
  
  for image_info, labels, bottom_up_data in data:
    if not bottom_up_data:
      continue
    image_filename = image_info[0]
    image_filepath = os.path.join(tmp_dir, prefix, image_filename)
    with tf.gfile.Open(image_filepath, "r") as f:
      encoded_image_data = f.read()
      height, width = image_info[1], image_info[2]
      if training:
        processed_labels = []
        for label in labels:
          label = label.replace('.', '').replace(',', '').replace("'", "").replace('"', '').replace('!', '')
          label = label.replace('&', 'and').replace('(', '').replace(")", "").replace('-', ' ').replace(';', '')
          label = label.replace('#', '').replace(':', '').replace('?', '')
          label = " ".join(label.split())  # replace multiple spaces
          label = label.lower()
          label_utf = label.encode("utf8")
          processed_labels.append(label_utf)
          if vocab_filename is None or vocab_symbolizer is None:
            label = [ord(c) for c in label] + eos_list
          else:
            label = vocab_symbolizer.encode(label) + eos_list
          '''
          print("Label")
          print(label)
          print("Object Features")
          print(bottom_up_data['features'])
          '''
          yield {
              "image/encoded": [encoded_image_data],
              "image/features": bottom_up_data['features'],
              "image/boxes": bottom_up_data['boxes'],
              "image/num_boxes": [bottom_up_data['num_boxes']],
              "image/format": ["jpeg"],
              "image/class/label": label,
              "image/class/label_image": [int(bottom_up_data['image_id'])],
              "image/height": [height],
              "image/width": [width]
          }
        train_caption_path = os.path.join(data_dir, "train_caption.txt")
        with open(train_caption_path, "a") as myfile:
          image_id = int(bottom_up_data['image_id'])
          myfile.write(str(image_id) + ':' + str(processed_labels) + '\n')
      else:
        tokenised_labels = []
        processed_labels = []
        for label in labels:
          label = label.replace('.', '').replace(',', '').replace("'", "").replace('"', '').replace('!', '')
          label = label.replace('&', 'and').replace('(', '').replace(")", "").replace('-', ' ').replace(';', '')
          label = label.replace('#', '').replace(':', '').replace('?', '')
          label = " ".join(label.split())  # replace multiple spaces
          label = label.lower()
          label_utf = label.encode("utf8")
          processed_labels.append(label_utf)
          if vocab_filename is None or vocab_symbolizer is None:
            label = [ord(c) for c in label] + eos_list
          else:
            # print(image_filepath + ':' + label)
            label = vocab_symbolizer.encode(label) + eos_list
          tokenised_labels.append(label)
        image_id += 1
        val_caption_path = os.path.join(data_dir, "val_caption.txt")
        with open(val_caption_path, "a") as myfile:
          myfile.write(str(image_id) + ':' + str(processed_labels) + '\n')

        val_image_path = os.path.join(data_dir, "val_image_id.txt")
        with open(val_image_path, "a") as myfile:
          myfile.write(str(bottom_up_data['image_id']) + '\n')
        yield {
          "image/encoded": [encoded_image_data],
          "image/features": bottom_up_data['features'],
          "image/boxes": bottom_up_data['boxes'],
          "image/num_boxes": [bottom_up_data['num_boxes']],
          "image/format": ["jpeg"],
          "image/class/label": tokenised_labels[0],
          "image/class/label_image": [image_id],
          "image/height": [height],
          "image/width": [width]
        }


@registry.register_problem
class ImageMsCocoCharacters(image_utils.Image2TextProblem):
  """MSCOCO, character level."""

  @property
  def is_character_level(self):
    return True

  @property
  def target_space_id(self):
    return problem.SpaceID.EN_CHR

  @property
  def train_shards(self):
    return 100

  @property
  def dev_shards(self):
    return 8

  def preprocess_example(self, example, mode, _):
    return imagenet.imagenet_preprocess_example(example, mode)

  def generator(self, data_dir, tmp_dir, is_training):
    if is_training:
      return mscoco_generator(data_dir, tmp_dir, True, 80000)
    else:
      return mscoco_generator(data_dir, tmp_dir, False, 40000)
    raise NotImplementedError()


@registry.register_problem
class ImageMsCocoTokens32k(ImageMsCocoCharacters):
  """MSCOCO, 8k tokens vocab."""

  @property
  def is_character_level(self):
    return False

  @property
  def targeted_vocab_size(self):
    #return 2**15  # 32768
    #return 14144  # 14144
    return 9996  # 14144
    #return 9828

  @property
  def target_space_id(self):
    return problem.SpaceID.EN_TOK

  @property
  def train_shards(self):
    return 50
#    return 1

  @property
  def dev_shards(self):
    return 12
#    return 1

  def generator(self, data_dir, tmp_dir, is_training):
    # We use the translate vocab file as the vocabulary for captions.
    # This requires having the vocab file present in the data_dir for the
    # generation pipeline to succeed.
    vocab_filename = "vocab.ende.%d" % self.targeted_vocab_size
    if is_training:
      return mscoco_generator(
          data_dir,
          tmp_dir,
          True,
          113287,
#          100,
          vocab_filename=vocab_filename)
    else:
      return mscoco_generator(
          data_dir,
          tmp_dir,
          False,
#          50000,
          100,
          vocab_filename=vocab_filename)

  def example_reading_spec_train(self):
    label_image = "image/class/label_image"
    image_feature_key = "image/features"
    image_boxes_key = "image/boxes"
    image_num_boxes_key = "image/num_boxes"

    data_fields, data_items_to_decoders = (
        super(ImageMsCocoTokens32k, self).example_reading_spec())

    data_fields[label_image] = tf.VarLenFeature(tf.int64)
    data_items_to_decoders[
      "infer_image"] = tf.contrib.slim.tfexample_decoder.Tensor(label_image)

    if use_bottom_up_features:
      data_fields[image_feature_key] = tf.VarLenFeature(tf.float32)
      data_items_to_decoders[
          "bottom_up_features"] = tf.contrib.slim.tfexample_decoder.Tensor(image_feature_key, shape = [-1,2048])

      data_fields[image_boxes_key] = tf.VarLenFeature(tf.float32)
      data_items_to_decoders[
          "object_boxes"] = tf.contrib.slim.tfexample_decoder.Tensor(image_boxes_key, shape = [-1,4])

      data_fields[image_num_boxes_key] = tf.VarLenFeature(tf.int64)
      #data_fields[image_num_boxes_key] = tf.FixedLenFeature([0],tf.int64)
      data_items_to_decoders[
          "object_num_boxes"] = tf.contrib.slim.tfexample_decoder.Tensor(image_num_boxes_key)
    return data_fields, data_items_to_decoders

  def example_reading_spec_val(self):
    label_image = "image/class/label_image"
    image_feature_key = "image/features"
    image_boxes_key = "image/boxes"
    image_num_boxes_key = "image/num_boxes"

    data_fields, data_items_to_decoders = (
        super(ImageMsCocoTokens32k, self).example_reading_spec())

    data_fields[label_image] = tf.VarLenFeature(tf.int64)
    data_items_to_decoders[
      "infer_image"] = tf.contrib.slim.tfexample_decoder.Tensor(label_image)

    if use_bottom_up_features:
      data_fields[image_feature_key] = tf.VarLenFeature(tf.float32)
      data_items_to_decoders[
          "bottom_up_features"] = tf.contrib.slim.tfexample_decoder.Tensor(image_feature_key, shape = [-1,2048])

      data_fields[image_boxes_key] = tf.VarLenFeature(tf.float32)
      data_items_to_decoders[
          "object_boxes"] = tf.contrib.slim.tfexample_decoder.Tensor(image_boxes_key, shape = [-1,4])

      data_fields[image_num_boxes_key] = tf.VarLenFeature(tf.int64)
      data_items_to_decoders[
          "object_num_boxes"] = tf.contrib.slim.tfexample_decoder.Tensor(image_num_boxes_key)
    return data_fields, data_items_to_decoders

  def dataset(self,
              mode,
              data_dir=None,
              num_threads=None,
              output_buffer_size=None,
              shuffle_files=None,
              repeat=None,
              hparams=None,
              preprocess=True,
              dataset_split=None,
              shard=None,
              partition_id=0,
              num_partitions=1):
    """Build a Dataset for this problem.

    Args:
      mode: tf.estimator.ModeKeys; determines which files to read from.
      data_dir: directory that contains data files.
      num_threads: int, number of threads to use for decode and preprocess
        Dataset.map calls.
      output_buffer_size: int, how many elements to prefetch at end of pipeline.
      shuffle_files: whether to shuffle input files. Default behavior (i.e. when
        shuffle_files=None) is to shuffle if mode == TRAIN.
      repeat: whether to repeat the Dataset. Default behavior is to repeat if
        mode == TRAIN.
      hparams: tf.contrib.training.HParams; hparams to be passed to
        Problem.preprocess_example and Problem.hparams. If None, will use a
        default set that is a no-op.
      preprocess: bool, whether to map the Dataset through
        Problem.preprocess_example.
      dataset_split: tf.estimator.ModeKeys + ["test"], which split to read data
        from (TRAIN:"-train", EVAL:"-dev", "test":"-test"). Defaults to mode.
      shard: int, if provided, will only read data from the specified shard.
      partition_id: integer - which partition of the dataset to read from
      num_partitions: how many partitions in the dataset

    Returns:
      Dataset containing dict<feature name, Tensor>.

    Raises:
      ValueError: if num_partitions is greater than the number of data files.
    """
    is_training = mode == tf.estimator.ModeKeys.TRAIN
    repeat = repeat or repeat is None and is_training
    shuffle_files = shuffle_files or shuffle_files is None and is_training

    dataset_split = dataset_split or mode
    assert data_dir

    if hparams is None:
      hparams = default_model_hparams()

    if not hasattr(hparams, "data_dir"):
      hparams.add_hparam("data_dir", data_dir)
    if not hparams.data_dir:
      hparams.data_dir = data_dir
    # Construct the Problem's hparams so that items within it are accessible
    _ = self.get_hparams(hparams)

    data_filepattern = self.filepattern(data_dir, dataset_split, shard=shard)
    tf.logging.info("Reading data files from %s", data_filepattern)
    data_files = tf.contrib.slim.parallel_reader.get_data_files(
        data_filepattern)

    # Functions used in dataset transforms below
    def _load_records(filename):
      # Load records from file with an 8MiB read buffer.
      return tf.data.TFRecordDataset(filename, buffer_size=8 * 1024 * 1024)

    def _preprocess(example):
      examples = self.preprocess_example(example, mode, hparams)
      if not isinstance(examples, tf.data.Dataset):
        examples = tf.data.Dataset.from_tensors(examples)
      return examples

    def _maybe_reverse_and_copy(example):
      self.maybe_reverse_features(example)
      self.maybe_copy_features(example)
      return example

    if len(data_files) < num_partitions:
      raise ValueError(
          "number of data files (%d) must be at least the number of hosts (%d)"
          % (len(data_files), num_partitions))
    data_files = [f for (i, f) in enumerate(data_files)
                  if i % num_partitions == partition_id]
    tf.logging.info(
        "partition: %d num_data_files: %d" % (partition_id, len(data_files)))
    if shuffle_files:
      random.shuffle(data_files)
    dataset = tf.data.Dataset.from_tensor_slices(tf.constant(data_files))

    if hasattr(tf.contrib.data, "parallel_interleave"):
      dataset = dataset.apply(
          tf.contrib.data.parallel_interleave(
              _load_records, sloppy=is_training, cycle_length=8))
    else:
      dataset = dataset.interleave(_load_records, cycle_length=8,
                                   block_length=16)

    if repeat:
      dataset = dataset.repeat()

    if is_training:
      dataset = dataset.map(self.decode_example, num_parallel_calls=num_threads)
    else:
      dataset = dataset.map(self.decode_example_val, num_parallel_calls=num_threads)

    if preprocess:
      if hasattr(tf.contrib.data, "parallel_interleave"):
        dataset = dataset.apply(
            tf.contrib.data.parallel_interleave(
                _preprocess, sloppy=is_training, cycle_length=8))
      else:
        dataset = dataset.interleave(_preprocess, cycle_length=8,
                                     block_length=16)
    dataset = dataset.map(
        _maybe_reverse_and_copy, num_parallel_calls=num_threads)

    if output_buffer_size:
      dataset = dataset.prefetch(output_buffer_size)

    return dataset

  def decode_example_val(self, serialized_example):
    """Return a dict of Tensors from a serialized tensorflow.Example."""
    data_fields, data_items_to_decoders = self.example_reading_spec_val()
    if data_items_to_decoders is None:
      data_items_to_decoders = {
          field: tf.contrib.slim.tfexample_decoder.Tensor(field)
          for field in data_fields
      }

    decoder = tf.contrib.slim.tfexample_decoder.TFExampleDecoder(
        data_fields, data_items_to_decoders)

    decode_items = list(data_items_to_decoders)
    decoded = decoder.decode(serialized_example, items=decode_items)
    return dict(zip(decode_items, decoded))

  def decode_example(self, serialized_example):
    """Return a dict of Tensors from a serialized tensorflow.Example."""
    data_fields, data_items_to_decoders = self.example_reading_spec_train()
    if data_items_to_decoders is None:
      data_items_to_decoders = {
          field: tf.contrib.slim.tfexample_decoder.Tensor(field)
          for field in data_fields
      }

    decoder = tf.contrib.slim.tfexample_decoder.TFExampleDecoder(
        data_fields, data_items_to_decoders)

    decode_items = list(data_items_to_decoders)
    decoded = decoder.decode(serialized_example, items=decode_items)
    return dict(zip(decode_items, decoded))


  def eval_metrics(self):
    return [
        metrics.Metrics.ACC, metrics.Metrics.ACC_TOP5,
        metrics.Metrics.ACC_PER_SEQ, metrics.Metrics.NEG_LOG_PERPLEXITY,
        metrics.Metrics.APPROX_BLEU , metrics.Metrics.ROUGE_2_F,
        metrics.Metrics.ROUGE_L_F, metrics.Metrics.BLEU_4,metrics.Metrics.CIDER
    ]

@registry.register_problem
class ImageTextMsCoco(ImageMsCocoTokens32k):
  """Problem for using MsCoco for generating images from text."""
  _MSCOCO_IMAGE_SIZE = 32

  def dataset_filename(self):
    return "image_ms_coco_tokens32k"

  def preprocess_example(self, example, mode, unused_hparams):
    example["inputs"] = image_utils.resize_by_area(
        example["inputs"], self._MSCOCO_IMAGE_SIZE)
    return example
