# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from itertools import islice

from scipy import sparse, io

import collections
import csv
import os
import modeling
import optimization
import tokenization
import tensorflow as tf

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", None, "The name of the task to train.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_string(
    "label_vocab", None,
    "The file which contains the class labels.")

flags.DEFINE_string(
    "train_file", None,
    "The file which contains the training snippets.")

flags.DEFINE_string(
    "train_labels", None,
    "The file which contains the training labels.")

flags.DEFINE_string(
    "train_adj", None,
    "The file which contains the training adjacency matrices.")

flags.DEFINE_string(
    "eval_file", None,
    "The file which contains the evaluation snippets.")

flags.DEFINE_string(
    "eval_labels", None,
    "The file which contains the evaluation labels.")

flags.DEFINE_string(
    "eval_adj", None,
    "The file which contains the evaluation adjacency matrices.")

flags.DEFINE_bool("clean_data", False, "Whether to generate a fresh batch of data.")

flags.DEFINE_integer(
    "max_nb_preds", 128,
    "Max number of predictions to generate.")

## Other parameters

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_bool(
    "sparse_adj", False,
    "Whether to run the model in inference mode on the test set.")

tf.flags.DEFINE_string(
    "adj_prefix", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")


class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text_a, adjacency, label=None):
    """Constructs a InputExample.
    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text_a = text_a
    self.label = label
    self.adjacency = adjacency


class PaddingInputExample(object):
  """Fake example so the num input examples is a multiple of the batch size.
  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.
  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_ids,
               adjacency,
               input_mask,
               segment_ids,
               label_id,
               is_real_example=True):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.adjacency = adjacency
    self.segment_ids = segment_ids
    self.label_id = label_id
    self.is_real_example = is_real_example


class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""

  def get_train_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the train set."""
    raise NotImplementedError()

  def get_dev_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the dev set."""
    raise NotImplementedError()

  def get_test_examples(self, data_dir):
    """Gets a collection of `InputExample`s for prediction."""
    raise NotImplementedError()

  def get_labels(self):
    """Gets the list of labels for this data set."""
    raise NotImplementedError()

  @classmethod
  def _read_tsv(cls, input_file, quotechar=None):
    """Reads a tab separated value file."""
    with tf.gfile.Open(input_file, "r") as f:
      reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
      lines = []
      for line in reader:
        lines.append(line)
      return lines


class MethodNamingProcessor(DataProcessor):
  """Processor for the Method Naming data set."""

  @classmethod
  def _read_adj(cls, input_file, from_line=0):
    with tf.gfile.Open(input_file, "r") as f:
      reader = csv.reader(f, delimiter=',')
      lines = []
      for row in islice(reader, from_line, from_line+FLAGS.max_seq_length):
        lines.append([int(r) for r in row])
      return lines

  @classmethod
  def _read_sparse_adj(cls, path, suffix, idx=0):
    fname = str(idx)+'_'+FLAGS.adj_prefix+suffix+'.mtx'
    m = io.mmread(os.path.join(path, 'adj', fname))
    return list(m.toarray()) 

  def get_train_examples(self, data_path, label_path):
    """See base class."""
    snippets = self._read_tsv(data_path)
    labels = self._read_tsv(label_path)
    return self._create_examples(snippets, labels, "train")

  def get_test_examples(self, data_path, label_path):
    """See base class."""
    snippets = self._read_tsv(data_path)
    labels = self._read_tsv(label_path)
    return self._create_examples(snippets, labels, "test")

  def get_labels(self, label_path):
    """See base class."""
    l = self._read_tsv(label_path)
    labels = [item for sublist in l for item in sublist]
    return labels

  def get_adjacency(self, adj_path, suffix="", idx=0):
    if FLAGS.sparse_adj:
      return self._read_sparse_adj(adj_path, suffix=suffix, idx=idx)
    else:
      return self._read_adj(adj_path, from_line=idx*(64+2))

  def _create_examples(self, lines, labels, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
    for (i, (line, label)) in enumerate(zip(lines, labels)):
      guid = "%s-%s" % (set_type, str(i))
      text  = tokenization.convert_to_unicode(line[0])
      label = tokenization.convert_to_unicode(label[0])

      adj_file = FLAGS.train_adj if set_type=="train" else FLAGS.eval_adj
      suffix = "_adj" if set_type=="train" else "_adj_val"
      adj = self.get_adjacency(adj_file, suffix=suffix, idx=i)

      examples.append(
          InputExample(guid=guid, text_a=text, adjacency=adj, label=label))
      #train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
      #file_based_convert_examples_to_features(
      #    examples, self.get_labels(FLAGS.label_vocab), FLAGS.max_seq_length, tokenizer, train_file)
    return examples

class VarNamingProcessor(DataProcessor):
  """Processor for the Method Naming data set."""

  @classmethod
  def _read_adj(cls, input_file, from_line=0):
    with tf.gfile.Open(input_file, "r") as f:
      reader = csv.reader(f, delimiter=',')
      lines = []
      for row in islice(reader, from_line, from_line+FLAGS.max_seq_length):
        lines.append([int(r) for r in row])
      return lines

  @classmethod
  def _read_sparse_adj(cls, path, suffix, idx=0):
    fname = str(idx)+'_'+FLAGS.adj_prefix+suffix+'.mtx'
    m = io.mmread(os.path.join(path, 'adj', fname))
    return list(m.toarray())

  def get_train_examples(self, data_path, label_path):
    """See base class."""
    snippets = self._read_tsv(data_path)
    labels = self._read_tsv(label_path)
    return self._create_examples(snippets, labels, "train")

  def get_test_examples(self, data_path, label_path):
    """See base class."""
    snippets = self._read_tsv(data_path)
    labels = self._read_tsv(label_path)
    return self._create_examples(snippets, labels, "test")

  def get_labels(self, label_path):
    """See base class."""
    l = self._read_tsv(label_path)
    labels = [item for sublist in l for item in sublist]
    return labels

  def get_adjacency(self, adj_path, suffix="", idx=0):
    if FLAGS.sparse_adj:
      return self._read_sparse_adj(adj_path, suffix=suffix, idx=idx)
    else:
      return self._read_adj(adj_path, from_line=idx*(64+2))

  def _create_examples(self, lines, labels, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
    for (i, (line, label)) in enumerate(zip(lines, labels)):
      if len(label) > 0:
        guid = "%s-%s" % (set_type, str(i))

        text  = tokenization.convert_to_unicode(line[0])
        label = tokenization.convert_to_unicode(label[0])
        
        adj_file = FLAGS.train_adj if set_type=="train" else FLAGS.eval_adj
        suffix = "_adj" if set_type=="train" else "_adj_val"
        adj = self.get_adjacency(adj_file, suffix=suffix, idx=i) 

        examples.append(
            InputExample(guid=guid, text_a=text, adjacency=adj, label=label))
    return examples


def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
  """Converts a single `InputExample` into a single `InputFeatures`."""

  if isinstance(example, PaddingInputExample):
    return InputFeatures(
        input_ids=[0] * max_seq_length,
        input_mask=[0] * max_seq_length,
        adjacency=[[0] * max_seq_length] * max_seq_length,
        segment_ids=[0] * max_seq_length,
        label_id=0,
        is_real_example=False)

  label_map = {}
  for (i, label) in enumerate(label_list):
    label_map[label] = i

  tokens_a = tokenizer.tokenize(example.text_a)

  # The convention in BERT is:
  # (a) For sequence pairs:
  #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
  #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
  # (b) For single sequences:
  #  tokens:   [CLS] the dog is hairy . [SEP]
  #  type_ids: 0     0   0   0  0     0 0
  #
  # Where "type_ids" are used to indicate whether this is the first
  # sequence or the second sequence. The embedding vectors for `type=0` and
  # `type=1` were learned during pre-training and are added to the wordpiece
  # embedding vector (and position vector). This is not *strictly* necessary
  # since the [SEP] token unambiguously separates the sequences, but it makes
  # it easier for the model to learn the concept of sequences.
  #
  # For classification tasks, the first vector (corresponding to [CLS]) is
  # used as the "sentence vector". Note that this only makes sense because
  # the entire model is fine-tuned.
  tokens = []
  segment_ids = []

  for token in tokens_a:
    tokens.append(token)
    segment_ids.append(0)

  input_ids = tokenizer.convert_tokens_to_ids(tokens)

  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
  input_mask = [1] * len(input_ids)

  # Zero-pad up to the sequence length.
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length

  split_label = tokenizer.tokenize(' '.join(example.label.split(',')))

  is_multilabel = len(split_label) > 1 
  if is_multilabel:
    split_label = tokenizer.convert_tokens_to_ids(split_label)
    label_id = []
    msk_count = 0
    for idx, token in enumerate(tokens_a):
      if token == "[MASK]":
        label_id.append(split_label[msk_count])
        msk_count += 1
      else:
        label_id.append(input_ids[idx])
    while len(label_id) < max_seq_length:
      label_id.append(0)
  else:
    label_id = label_map[example.label]

  if ex_index < 5:
    tf.logging.info("*** Example ***")
    tf.logging.info("guid: %s" % (example.guid))
    tf.logging.info("tokens: %s" % " ".join(
        [tokenization.printable_text(x) for x in tokens]))
    tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
    if is_multilabel:
      tf.logging.info("label: {} (ids = {})".format(example.label, label_id))
    else:
      tf.logging.info("label: %s (id = %d)" % (example.label, label_id))
    #for v in example.adjacency:
    #  tf.logging.info("adjacency: %s" % " ".join([str(x) for x in v]))

  feature = InputFeatures(
      input_ids=input_ids,
      adjacency=example.adjacency,
      input_mask=input_mask,
      segment_ids=segment_ids,
      label_id=label_id,
      is_real_example=True)
  return feature

def file_based_convert_examples_to_features(
    examples, label_list, max_seq_length, tokenizer, output_file):
  """Convert a set of `InputExample`s to a TFRecord file."""

  writer = tf.python_io.TFRecordWriter(output_file)

  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    feature = convert_single_example(ex_index, example, label_list,
                                     max_seq_length, tokenizer)

    def create_int_feature(values):
      f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
      return f

    adjacency = feature.adjacency

    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["adjacency"] = create_int_feature([item for sublist in adjacency for item in sublist])
    features["input_mask"] = create_int_feature(feature.input_mask)
    features["segment_ids"] = create_int_feature(feature.segment_ids)
    if FLAGS.task_name == 'varname':
      features["label_ids"] = create_int_feature(feature.label_id)
    else: 
      features["label_ids"] = create_int_feature([feature.label_id])
    features["is_real_example"] = create_int_feature(
        [int(feature.is_real_example)])
    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())
  writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  if FLAGS.task_name=='varname':
    lids = tf.FixedLenFeature([seq_length], tf.int64)
  else:
    lids = tf.FixedLenFeature([], tf.int64)
  name_to_features = {
      "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "adjacency": tf.FixedLenFeature([seq_length*seq_length], tf.int64),
      "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
      "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "label_ids": lids, 
      "is_real_example": tf.FixedLenFeature([], tf.int64),
  }

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.to_int32(t)
      example[name] = t

    return example

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.TFRecordDataset(input_file)
    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))

    return d

  return input_fn


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()


def create_model(bert_config, is_training, input_ids, input_mask, adj_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):
  """Creates a classification model."""
  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      adj_mask=adj_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)

  is_multilabel = FLAGS.task_name == 'varname'
  # In the demo, we are doing a simple classification task on the entire
  # segment.
  #
  # If you want to use the token-level output, use model.get_sequence_output()
  # instead.
  if is_multilabel:
    output_layer = model.get_sequence_output()
    final_hidden_shape = modeling.get_shape_list(output_layer, expected_rank=3)
    batch_size = final_hidden_shape[0]
    seq_length = final_hidden_shape[1]
    hidden_size = final_hidden_shape[2]
  else:
    output_layer = model.get_pooled_output()
    hidden_size = output_layer.shape[-1].value
  
  output_weights = tf.get_variable(
      "output_weights", [num_labels, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
      "output_bias", [num_labels], initializer=tf.zeros_initializer())

  with tf.variable_scope("loss"):
    if is_training:
      # I.e., 0.1 dropout
      output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
    if is_multilabel:
      output_layer = tf.reshape(output_layer, [batch_size*seq_length, hidden_size])
    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)

    if is_multilabel:
      logits = tf.reshape(logits, [batch_size, seq_length, num_labels])
      #logits = tf.transpose(logits, [2, 0, 1])
      #logits = tf.reshape(logits, [16, FLAGS.max_seq_length, num_labels])
    probabilities = tf.nn.softmax(logits, axis=-1)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)

    if is_multilabel:
      loss = tf.reduce_sum(per_example_loss)
    else:
      loss = tf.reduce_mean(per_example_loss)

    attention_output = model.get_attention_output()

    return (loss, per_example_loss, logits, probabilities, attention_output)


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_ids = features["label_ids"]
    adjacency = features["adjacency"]
    
    is_real_example = None
    if "is_real_example" in features:
      is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
    else:
      is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    (total_loss, per_example_loss, logits, probabilities, attention_output) = create_model(
        bert_config, is_training, input_ids, input_mask, adjacency, segment_ids, label_ids,
        num_labels, use_one_hot_embeddings)

    #logits = tf.unstack(logits, axis=0)
    #probabilities = tf.nn.softmax(logits, axis=-1)

    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:

      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn)
    elif mode == tf.estimator.ModeKeys.EVAL:

      def metric_fn(per_example_loss, label_ids, logits, is_real_example):
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        accuracy = tf.metrics.accuracy(
            labels=label_ids, predictions=predictions, weights=is_real_example)
        loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
        return {
            "eval_accuracy": accuracy,
            "eval_loss": loss,
        }

      eval_metrics = (metric_fn,
                      [per_example_loss, label_ids, logits, is_real_example])
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metrics=eval_metrics,
          scaffold_fn=scaffold_fn)
    else:
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          predictions={
            "probabilities": probabilities,
            "attention_output": attention_output
          },
          scaffold_fn=scaffold_fn)
    return output_spec

  return model_fn


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  processors = {
      "methodname": MethodNamingProcessor,
      "varname": VarNamingProcessor,
  }

  tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                FLAGS.init_checkpoint)

  if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
    raise ValueError(
        "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  tf.gfile.MakeDirs(FLAGS.output_dir)

  task_name = FLAGS.task_name.lower()

  if task_name not in processors:
    raise ValueError("Task not found: %s" % (task_name))

  processor = processors[task_name]()

  label_list = processor.get_labels(FLAGS.label_vocab)

  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))

  train_examples = None
  num_train_steps = None
  num_warmup_steps = None
  if FLAGS.do_train:
    train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
    if FLAGS.clean_data:
      train_examples = processor.get_train_examples(FLAGS.train_file, FLAGS.train_labels)
      num_train_steps = int(
          len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
      file_based_convert_examples_to_features(
          train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)
    else:
      num_train_steps = FLAGS.num_train_epochs
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

  model_fn = model_fn_builder(
      bert_config=bert_config,
      num_labels=len(label_list),
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size,
      predict_batch_size=FLAGS.predict_batch_size)

  if FLAGS.do_train:

    tf.logging.info("***** Running training *****")
    #tf.logging.info("  Num examples = %d", len(train_examples))
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    tf.logging.info("  Num steps = %d", num_train_steps)
    train_input_fn = file_based_input_fn_builder(
        input_file=train_file,
        seq_length=FLAGS.max_seq_length,
        is_training=True,
        drop_remainder=True)
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
 
  if FLAGS.do_eval:
    eval_examples = processor.get_test_examples(FLAGS.eval_file, FLAGS.eval_labels)
    num_actual_eval_examples = len(eval_examples)
    if FLAGS.use_tpu:
      # TPU requires a fixed batch size for all batches, therefore the number
      # of examples must be a multiple of the batch size, or else examples
      # will get dropped. So we pad with fake examples which are ignored
      # later on. These do NOT count towards the metric (all tf.metrics
      # support a per-instance weight, and these get a weight of 0.0).
      while len(eval_examples) % FLAGS.eval_batch_size != 0:
        eval_examples.append(PaddingInputExample())

    eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
    file_based_convert_examples_to_features(
        eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file)

    tf.logging.info("***** Running evaluation *****")
    tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                    len(eval_examples), num_actual_eval_examples,
                    len(eval_examples) - num_actual_eval_examples)
    tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

    # This tells the estimator to run through the entire set.
    eval_steps = None

    eval_drop_remainder = True if FLAGS.use_tpu else False
    eval_input_fn = file_based_input_fn_builder(
        input_file=eval_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=eval_drop_remainder)

    result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)

    output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
    with tf.gfile.GFile(output_eval_file, "w") as writer:
      tf.logging.info("***** Eval results *****")
      for key in sorted(result.keys()):
        tf.logging.info("  %s = %s", key, str(result[key]))
        writer.write("%s = %s\n" % (key, str(result[key])))

  if FLAGS.do_predict:
    predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
    if FLAGS.clean_data:
      predict_examples = processor.get_test_examples(FLAGS.eval_file, FLAGS.eval_labels)
      num_actual_predict_examples = len(predict_examples)
      file_based_convert_examples_to_features(predict_examples, label_list,
                                              FLAGS.max_seq_length, tokenizer,
                                              predict_file)
    else:
      num_actual_predict_examples = FLAGS.max_nb_preds
    if FLAGS.use_tpu:
      # TPU requires a fixed batch size for all batches, therefore the number
      # of examples must be a multiple of the batch size, or else examples
      # will get dropped. So we pad with fake examples which are ignored
      # later on.
      while len(predict_examples) % FLAGS.predict_batch_size != 0:
        predict_examples.append(PaddingInputExample())


    tf.logging.info("***** Running prediction*****")
    #tf.logging.info("  Num examples = %d (%d actual, %d padding)",
    #                len(predict_examples), num_actual_predict_examples,
    #                len(predict_examples) - num_actual_predict_examples)
    tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

    predict_drop_remainder = True if FLAGS.use_tpu else False
    predict_input_fn = file_based_input_fn_builder(
        input_file=predict_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=predict_drop_remainder)

    result = estimator.predict(input_fn=predict_input_fn)

    output_predict_file = os.path.join(FLAGS.output_dir, "test_results.tsv")
    with tf.gfile.GFile(output_predict_file, "w") as writer:
      num_written_lines = 0
      tf.logging.info("***** Predict results *****")
      for (i, prediction) in enumerate(result):
        probabilities = prediction["probabilities"]
        if task_name=='varname':
          for class_prob in probabilities: 
            out_line = [] 
            for prob in class_prob:
              out_line.append(str(prob))
            writer.write('\t'.join(out_line)+"\n")
          if i >= num_actual_predict_examples:
            break
        else:        
          output_line = "\t".join(
               str(class_probability)
               for class_probability in probabilities) + "\n"
          writer.write(output_line)
        num_written_lines += 1
    #assert num_written_lines == num_actual_predict_examples

    output_predict_file = os.path.join(FLAGS.output_dir, "attention_results.tsv")
    with tf.gfile.GFile(output_predict_file, "w") as writer:
      num_written_lines = 0
      tf.logging.info("***** Attention results *****")
      for (i, prediction) in enumerate(result):
        att = prediction["attention_output"]
        for i in range(12):
          for j in range(64):
            for k in range(64):
              writer.write("%s " % str(att[i][j][k]))
          writer.write("\n")
        break
        num_written_lines += 1


if __name__ == "__main__":
  flags.mark_flag_as_required("data_dir")
  flags.mark_flag_as_required("task_name")
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("label_vocab")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")
  tf.app.run()
