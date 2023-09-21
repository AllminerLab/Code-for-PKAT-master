# -*- coding: utf-8 -*-
""" 
@Time    : 2023/3/27 16:35
@Author  : ONER
@FileName: run_bert.py
@SoftWare: PyCharm
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from scipy.io import savemat

from utility import optimization, modeling

import tensorflow as tf
import numpy as np
import sys
import pickle
from utility.parser import parse_args

gpus = tf.config.experimental.list_physical_devices('GPU')

class EvalHooks(tf.train.SessionRunHook):
    def __init__(self):
        tf.logging.info('run init')

    def begin(self):
        self.valid_user = 0.0

        self.ndcg_1 = 0.0
        self.hit_1 = 0.0
        self.ndcg_5 = 0.0
        self.hit_5 = 0.0
        self.ndcg_10 = 0.0
        self.hit_10 = 0.0
        self.ap = 0.0

        self.h = np.zeros(shape=(0, 64))
        self.info = np.zeros(shape=(0, 1))

        np.random.seed(12345)

        self.vocab = None

        args = parse_args()
        user_history_filename = args.data_path + args.dataset + '/' + args.dataset + '.his'
        vocab_filename = args.data_path + args.dataset + '/' + args.dataset + '.vocab'
        if user_history_filename is not None:
            print('load user history from :' + user_history_filename)
            with open(user_history_filename, 'rb') as input_file:
                self.user_history = pickle.load(input_file)

        if vocab_filename is not None:
            print('load vocab from :' + vocab_filename)
            with open(vocab_filename, 'rb') as input_file:
                self.vocab = pickle.load(input_file)

            # Counter({'item_10': 3406, 'item_150': 2981, ...})
            keys = self.vocab.counter.keys()
            values = self.vocab.counter.values()
            self.ids = self.vocab.convert_tokens_to_ids(keys)
            # normalize
            # print(values)
            sum_value = np.sum([x for x in values])
            # print(sum_value)
            self.probability = [value / sum_value for value in values]

    def end(self, session):
        print(
            "ndcg@1:{}, hit@1:{}， ndcg@5:{}, hit@5:{}, ndcg@10:{}, hit@10:{}, ap:{}, valid_user:{}".
            format(self.ndcg_1 / self.valid_user, self.hit_1 / self.valid_user,
                   self.ndcg_5 / self.valid_user, self.hit_5 / self.valid_user,
                   self.ndcg_10 / self.valid_user,
                   self.hit_10 / self.valid_user, self.ap / self.valid_user,
                   self.valid_user))

        print(self.h)
        print(self.h.shape)
        print(self.info.shape)
        print(self.info)
        sum = np.append(self.info, self.h, axis=1)
        print(sum.shape)
        print(sum)
        sum = sum[np.lexsort(sum[:, ::-1].T)]
        print(sum)
        self.h = sum[:, 1:]
        print(self.h.shape)
        print(self.h)

        inp = np.mat(self.h)
        print('input matrix is:')
        print(inp, inp.shape)
        save_dict = {'name': 'matrix', 'data': inp}
        #  test.mat是保存路径，save_dict必须是dict类型，他就这么定义的！
        savemat('amazon-book_20.mat', save_dict)

    def before_run(self, run_context):
        #tf.logging.info('run before run')
        #print('run before_run')
        variables = tf.get_collection('eval_sp')
        return tf.train.SessionRunArgs(variables)

    def after_run(self, run_context, run_values):
        #tf.logging.info('run after run')
        #print('run after run')
        max_predictions_per_seq = 40
        masked_lm_log_probs, input_ids, masked_lm_ids, info, h = run_values.results

        self.h = np.append(self.h, h, axis=0)
        # self.h = h
        # self.info = info
        self.info = np.append(self.info, info, axis=0)


        masked_lm_log_probs = masked_lm_log_probs.reshape(
            (-1, max_predictions_per_seq, masked_lm_log_probs.shape[1]))
#         print("loss value:", masked_lm_log_probs.shape, input_ids.shape,
#               masked_lm_ids.shape, info.shape)

        for idx in range(len(input_ids)):
            rated = set(input_ids[idx])
            rated.add(0)
            rated.add(masked_lm_ids[idx][0])
            map(lambda x: rated.add(x),
                self.user_history["user_" + str(info[idx][0])][0])
            item_idx = [masked_lm_ids[idx][0]]
            # here we need more consideration
            masked_lm_log_probs_elem = masked_lm_log_probs[idx, 0]
            size_of_prob = len(self.ids) + 1  # len(masked_lm_log_probs_elem)
            use_pop_random = True
            if use_pop_random:
                if self.vocab is not None:
                    while len(item_idx) < 101:
                        sampled_ids = np.random.choice(self.ids, 101, replace=False, p=self.probability)
                        sampled_ids = [x for x in sampled_ids if x not in rated and x not in item_idx]
                        item_idx.extend(sampled_ids[:])
                    item_idx = item_idx[:101]
            else:
                # print("evaluation random -> ")
                for _ in range(100):
                    t = np.random.randint(1, size_of_prob)
                    while t in rated:
                        t = np.random.randint(1, size_of_prob)
                    item_idx.append(t)

            predictions = -masked_lm_log_probs_elem[item_idx]
            rank = predictions.argsort().argsort()[0]

            self.valid_user += 1

            if self.valid_user % 100 == 0:
                print('.', end='')
                sys.stdout.flush()

            if rank < 1:
                self.ndcg_1 += 1
                self.hit_1 += 1
            if rank < 5:
                self.ndcg_5 += 1 / np.log2(rank + 2)
                self.hit_5 += 1
            if rank < 10:
                self.ndcg_10 += 1 / np.log2(rank + 2)
                self.hit_10 += 1

            self.ap += 1.0 / (rank + 1)


def input_fn_builder(input_files,
                     max_seq_length,
                     max_predictions_per_seq,
                     is_training,
                     num_cpu_threads=4):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        name_to_features = {
            "info":
            tf.FixedLenFeature([1], tf.int64),  #[user]
            "input_ids":
            tf.FixedLenFeature([max_seq_length], tf.int64),
            "input_mask":
            tf.FixedLenFeature([max_seq_length], tf.int64),
            "masked_lm_positions":
            tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
            "masked_lm_ids":
            tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
            "masked_lm_weights":
            tf.FixedLenFeature([max_predictions_per_seq], tf.float32)
        }

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        if is_training:
            d = tf.data.TFRecordDataset(input_files)
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

            # `cycle_length` is the number of parallel files that get read.
            #cycle_length = min(num_cpu_threads, len(input_files))

            # `sloppy` mode means that the interleaving is not exact. This adds
            # even more randomness to the training pipeline.
            #d = d.apply(
            #    tf.contrib.data.parallel_interleave(
            #        tf.data.TFRecordDataset,
            #        sloppy=is_training,
            #        cycle_length=cycle_length))
            #d = d.shuffle(buffer_size=100)
        else:
            d = tf.data.TFRecordDataset(input_files)


        d = d.map(
            lambda record: _decode_record(record, name_to_features),
            num_parallel_calls=num_cpu_threads)
        d = d.batch(batch_size=batch_size)
        return d

    return input_fn


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

def gather_indexes(sequence_tensor, positions):
    """Gathers the vectors at the specific positions over a minibatch."""
    sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
    batch_size = sequence_shape[0]
    seq_length = sequence_shape[1]
    width = sequence_shape[2]

    flat_offsets = tf.reshape(
        tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
    flat_positions = tf.reshape(positions + flat_offsets, [-1])
    flat_sequence_tensor = tf.reshape(sequence_tensor,
                                      [batch_size * seq_length, width])
    output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
    return output_tensor



def get_masked_lm_output(bert_config, input_tensor, output_weights, positions,
                         label_ids, label_weights):
    """Get loss and log probs for the masked LM."""
    # [batch_size*label_size, dim]
    input_tensor = gather_indexes(input_tensor, positions)

    with tf.variable_scope("cls/predictions"):
        # We apply one more non-linear transformation before the output layer.
        # This matrix is not used after pre-training.
        with tf.variable_scope("transform"):
            input_tensor = tf.layers.dense(
                input_tensor,
                units=bert_config.hidden_size,
                activation=modeling.get_activation(bert_config.hidden_act),
                kernel_initializer=modeling.create_initializer(
                    bert_config.initializer_range))
            input_tensor = modeling.layer_norm(input_tensor)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        output_bias = tf.get_variable(
            "output_bias",
            shape=[output_weights.shape[0]],
            initializer=tf.zeros_initializer())
        logits = tf.matmul(input_tensor, output_weights, transpose_b=True) #对第二个参数b进行转置
        logits = tf.nn.bias_add(logits, output_bias)
        # logits, (bs*label_size, vocab_size)
        log_probs = tf.nn.log_softmax(logits, -1)

        label_ids = tf.reshape(label_ids, [-1])
        label_weights = tf.reshape(label_weights, [-1])

        one_hot_labels = tf.one_hot(
            label_ids, depth=output_weights.shape[0], dtype=tf.float32)

        # The `positions` tensor might be zero-padded (if the sequence is too
        # short to have the maximum number of predictions). The `label_weights`
        # tensor has a value of 1.0 for every real prediction and 0.0 for the
        # padding predictions.
        per_example_loss = -tf.reduce_sum(
            log_probs * one_hot_labels, axis=[-1])
        numerator = tf.reduce_sum(label_weights * per_example_loss)
        denominator = tf.reduce_sum(label_weights) + 1e-5
        loss = numerator / denominator

    return (loss, per_example_loss, log_probs)



def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings, item_size):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name,
                                                         features[name].shape))

        info = features["info"]
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        masked_lm_positions = features["masked_lm_positions"]
        masked_lm_ids = features["masked_lm_ids"]
        masked_lm_weights = features["masked_lm_weights"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        model = modeling.BertModel(
            config=bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=None,
            use_one_hot_embeddings=use_one_hot_embeddings)

        #         all_user_and_item = model.get_embedding_table()
        #         item_ids = [i for i in range(0, item_size + 1)]
        #         softmax_output_embedding = tf.nn.embedding_lookup(all_user_and_item, item_ids)

        # 这里可以得到model.get_sequence_output()的size为[batch_size, seq_length, hidden_size]
        # 此处的理解为对于eval来说，batch_size代表不同user的序列，seq_length是历史item长度，hidden_size是dim
        # 可以直接返回 model.get_sequence_output()的最后一个位置作为user的附加特征 即最后一个[MASK]

        h = model.get_sequence_output()
        print(h)
        h = gather_indexes(h, masked_lm_positions)  # 256*40,dim
        # h = tf.matmul(masked_lm_weights, h)
        h = h[::40]
        tf.logging.info("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        tf.logging.info(h.shape)

        (masked_lm_loss,
         masked_lm_example_loss, masked_lm_log_probs) = get_masked_lm_output(
            bert_config,
            model.get_sequence_output(),
            model.get_embedding_table(), masked_lm_positions, masked_lm_ids,
            masked_lm_weights)

        total_loss = masked_lm_loss

        tvars = tf.trainable_variables()

        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(
                tvars, init_checkpoint)
            if use_tpu:

                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint,
                                                  assignment_map)
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
            train_op = optimization.create_optimizer(total_loss, learning_rate,
                                                     num_train_steps,
                                                     num_warmup_steps, use_tpu)

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold=scaffold_fn)

        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(masked_lm_example_loss, masked_lm_log_probs,
                          masked_lm_ids, masked_lm_weights):
                """Computes the loss and accuracy of the model."""
                masked_lm_log_probs = tf.reshape(
                    masked_lm_log_probs, [-1, masked_lm_log_probs.shape[-1]])
                masked_lm_predictions = tf.argmax(
                    masked_lm_log_probs, axis=-1, output_type=tf.int32)
                masked_lm_example_loss = tf.reshape(masked_lm_example_loss,
                                                    [-1])
                masked_lm_ids = tf.reshape(masked_lm_ids, [-1])
                masked_lm_weights = tf.reshape(masked_lm_weights, [-1])
                masked_lm_accuracy = tf.metrics.accuracy(
                    labels=masked_lm_ids,
                    predictions=masked_lm_predictions,
                    weights=masked_lm_weights)
                masked_lm_mean_loss = tf.metrics.mean(
                    values=masked_lm_example_loss, weights=masked_lm_weights)

                return {
                    "masked_lm_accuracy": masked_lm_accuracy,
                    "masked_lm_loss": masked_lm_mean_loss,
                }

            tf.add_to_collection('eval_sp', masked_lm_log_probs)
            tf.add_to_collection('eval_sp', input_ids)
            tf.add_to_collection('eval_sp', masked_lm_ids)
            tf.add_to_collection('eval_sp', info)
            tf.add_to_collection('eval_sp', h)

            eval_metrics = metric_fn(masked_lm_example_loss,
                                     masked_lm_log_probs, masked_lm_ids,
                                     masked_lm_weights)
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metric_ops=eval_metrics,
                scaffold=scaffold_fn)
        else:
            raise ValueError("Only TRAIN and EVAL modes are supported: %s" %
                             (mode))

        return output_spec

    return model_fn


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    args = parse_args()
    bert_config_file = 'bert_train/bert_config_amazon-book_64.json'
    checkpointDir = args.dataset + 'BERT_checkpoint'
    print('checkpointDir:', checkpointDir)

    bert_config = modeling.BertConfig.from_json_file(bert_config_file)

    tf.gfile.MakeDirs(checkpointDir)
    train_input_file = args.data_path + args.dataset + '/' + args.dataset + '.train.tfrecord'
    test_input_file = args.data_path + args.dataset + '/' + args.dataset + '.test.tfrecord'

    train_input_files = []
    for input_pattern in train_input_file.split(","):
        train_input_files.extend(tf.gfile.Glob(input_pattern))

    test_input_files = []
    for input_pattern in test_input_file.split(","):
        test_input_files.extend(tf.gfile.Glob(input_pattern))

    tf.logging.info("*** train Input Files ***")
    for input_file in train_input_files:
        tf.logging.info("  %s" % input_file)

    tf.logging.info("*** test Input Files ***")
    for input_file in train_input_files:
        tf.logging.info("  %s" % input_file)

    tpu_cluster_resolver = None
    save_checkpoints_steps = 1000 #How often to save the model checkpoint.

    # is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.estimator.RunConfig(
        model_dir=checkpointDir,
        save_checkpoints_steps=save_checkpoints_steps)
    vocab_filename=args.data_path + args.dataset + '/' + args.dataset + '.vocab'
    if vocab_filename is not None:
        with open(vocab_filename, 'rb') as input_file:
            vocab = pickle.load(input_file)
    item_size = len(vocab.counter)

    learning_rate=1e-4
    # num_train_steps=400000
    num_train_steps = 20000
    num_warmup_steps=100
    print("item_size:")
    print(item_size)


    model_fn = model_fn_builder(
        bert_config=bert_config,
        init_checkpoint=None,
        learning_rate=learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=False,
        use_one_hot_embeddings=False,
        item_size=item_size)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config,
        params={
            "batch_size": 64
        })

    batch_size = 64
    do_train = True
    max_seq_length = 50
    max_predictions_per_seq = 10

    if do_train:
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Batch size = %d", batch_size)
        train_input_fn = input_fn_builder(
            input_files=train_input_files,
            max_seq_length=max_seq_length,
            max_predictions_per_seq=max_predictions_per_seq,
            is_training=True)
        estimator.train(
            input_fn=train_input_fn, max_steps=num_train_steps)

    do_eval = True
    if do_eval:
        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Batch size = %d", batch_size)

        eval_input_fn = input_fn_builder(
            input_files=test_input_files,
            max_seq_length=max_seq_length,
            max_predictions_per_seq=max_predictions_per_seq,
            is_training=False)

        #tf.logging.info('special eval ops:', special_eval_ops)
        result = estimator.evaluate(
            input_fn=eval_input_fn,
            steps=None,
            hooks=[EvalHooks()])

        output_eval_file = os.path.join(checkpointDir,
                                        "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            tf.logging.info(bert_config.to_json_string())
            writer.write(bert_config.to_json_string()+'\n')
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))


if __name__ == "__main__":
    tf.app.run()