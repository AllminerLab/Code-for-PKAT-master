import collections
import multiprocessing
import pickle
import random

import numpy as np
import random as rd
import tensorflow as tf

import time
import six

def create_training_instances(all_documents_raw,
                              max_seq_length,
                              dupe_factor,
                              short_seq_prob,
                              masked_lm_prob,
                              max_predictions_per_seq,
                              rng,
                              vocab,
                              mask_prob,
                              prop_sliding_window,
                              pool_size,
                              force_last=False):
    """Create `TrainingInstance`s from raw text."""
    all_documents = {}

    if force_last:
        max_num_tokens = max_seq_length
        for user, item_seq in all_documents_raw.items():
            if len(item_seq) == 0:
                print("got empty seq:" + user)
                continue
            all_documents[user] = [item_seq[-max_num_tokens:]]
    else:
        max_num_tokens = max_seq_length  # we need two sentence

        sliding_step = (int)(
            prop_sliding_window *
            max_num_tokens) if prop_sliding_window != -1.0 else max_num_tokens
        for user, item_seq in all_documents_raw.items():
            if len(item_seq) == 0:
                print("got empty seq:" + user)
                continue

            # todo: add slide
            if len(item_seq) <= max_num_tokens:
                all_documents[user] = [item_seq]
            else:
                beg_idx = list(range(len(item_seq) - max_num_tokens, 0, -sliding_step))
                beg_idx.append(0)
                all_documents[user] = [item_seq[i:i + max_num_tokens] for i in beg_idx[::-1]]

    instances = []
    if force_last:   #test集
        for user in all_documents:
            instances.extend(
                create_instances_from_document_test(
                    all_documents, user, max_seq_length))
        print("num of instance:{}".format(len(instances)))
    else:
        start_time = time.perf_counter()
        pool = multiprocessing.Pool(processes=pool_size)
        instances = []
        print("document num: {}".format(len(all_documents)))

        def log_result(result):
            print("callback function result type: {}, size: {} ".format(type(result), len(result)))
            instances.extend(result)

        for step in range(dupe_factor):
            pool.apply_async(
                create_instances_threading, args=(
                    all_documents, user, max_seq_length, short_seq_prob,
                    masked_lm_prob, max_predictions_per_seq, vocab, random.Random(random.randint(1, 10000)),
                    mask_prob, step), callback=log_result)
        pool.close()
        pool.join()

        for user in all_documents:
            instances.extend(  #将最后一个item转化为[mask]
                mask_last(
                    all_documents, user, max_seq_length, short_seq_prob,
                    masked_lm_prob, max_predictions_per_seq, vocab, rng))

        print("num of instance:{}; time:{}".format(len(instances), time.perf_counter() - start_time))
    rng.shuffle(instances)
    return instances


def mask_last(
        all_documents, user, max_seq_length, short_seq_prob, masked_lm_prob,
        max_predictions_per_seq, vocab, rng):
    """Creates `TrainingInstance`s for a single document."""
    document = all_documents[user]
    max_num_tokens = max_seq_length

    instances = []
    info = [int(user.split("_")[1])]
    vocab_items = vocab.get_items()

    for tokens in document:
        assert len(tokens) >= 1 and len(tokens) <= max_num_tokens

        (tokens, masked_lm_positions,
         masked_lm_labels) = create_masked_lm_predictions_force_last(tokens) #最后一个item转化为[mask]
        instance = TrainingInstance(
            info=info,
            tokens=tokens,
            masked_lm_positions=masked_lm_positions,
            masked_lm_labels=masked_lm_labels)
        instances.append(instance)

    return instances

def create_instances_threading(all_documents, user, max_seq_length, short_seq_prob,
                               masked_lm_prob, max_predictions_per_seq, vocab, rng,
                               mask_prob, step):
    cnt = 0;
    start_time = time.perf_counter()
    instances = []
    for user in all_documents:
        cnt += 1;
        if cnt % 1000 == 0:
            print("step: {}, name: {}, step: {}, time: {}".format(step, multiprocessing.current_process().name, cnt,
                                                                  time.perf_counter() - start_time))
            start_time = time.perf_counter()
        instances.extend(create_instances_from_document_train(
            all_documents, user, max_seq_length, short_seq_prob,
            masked_lm_prob, max_predictions_per_seq, vocab, rng,
            mask_prob))

    return instances


def create_instances_from_document_train(
        all_documents, user, max_seq_length, short_seq_prob, masked_lm_prob,
        max_predictions_per_seq, vocab, rng, mask_prob):
    """Creates `TrainingInstance`s for a single document."""
    document = all_documents[user]

    max_num_tokens = max_seq_length

    instances = []
    info = [int(user.split("_")[1])]
    vocab_items = vocab.get_items() #['item_1', 'item_2',...]

    for tokens in document:
        assert len(tokens) >= 1 and len(tokens) <= max_num_tokens

        (tokens, masked_lm_positions,
         masked_lm_labels) = create_masked_lm_predictions(
            tokens, masked_lm_prob, max_predictions_per_seq,
            vocab_items, rng, mask_prob)
        instance = TrainingInstance(
            info=info,
            tokens=tokens,
            masked_lm_positions=masked_lm_positions,
            masked_lm_labels=masked_lm_labels)
        instances.append(instance)

    return instances

def create_masked_lm_predictions(tokens, masked_lm_prob,
                                 max_predictions_per_seq, vocab_words, rng,
                                 mask_prob):
    """Creates the predictions for the masked LM objective."""

    cand_indexes = []
    for (i, token) in enumerate(tokens):
        if token not in vocab_words:
            continue
        cand_indexes.append(i)
    #[0, 1, 2, 3, 4, ..] -> [52, 35, 25, 30,...]
    rng.shuffle(cand_indexes)

    output_tokens = list(tokens)

    num_to_predict = min(max_predictions_per_seq,
                         max(1, int(round(len(tokens) * masked_lm_prob))))

    masked_lms = []
    covered_indexes = set()
    for index in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        if index in covered_indexes:
            continue
        covered_indexes.add(index)

        masked_token = None
        # 80% of the time, replace with [MASK]
        if rng.random() < mask_prob:
            masked_token = "[MASK]"
        else:
            # 10% of the time, keep original
            if rng.random() < 0.5:
                masked_token = tokens[index]
            # 10% of the time, replace with random word
            else:
                # masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]
                masked_token = rng.choice(vocab_words)

        output_tokens[index] = masked_token

        masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))

    masked_lms = sorted(masked_lms, key=lambda x: x.index)

    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)

    return (output_tokens, masked_lm_positions, masked_lm_labels)

MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])

def create_instances_from_document_test(all_documents, user, max_seq_length):
    """Creates `TrainingInstance`s for a single document."""
    document = all_documents[user]
    max_num_tokens = max_seq_length

    assert len(document) == 1 and len(document[0]) <= max_num_tokens

    tokens = document[0]
    assert len(tokens) >= 1

    (tokens, masked_lm_positions,
     masked_lm_labels) = create_masked_lm_predictions_force_last(tokens) #将最后一个真实值改为[mask]

    info = [int(user.split("_")[1])]
    instance = TrainingInstance(
        info=info,
        tokens=tokens,
        masked_lm_positions=masked_lm_positions,
        masked_lm_labels=masked_lm_labels)

    return [instance]

def create_masked_lm_predictions_force_last(tokens):
    """Creates the predictions for the masked LM objective."""

    last_index = -1
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[PAD]" or token == '[NO_USE]':
            continue
        last_index = i

    assert last_index > 0

    output_tokens = list(tokens)
    output_tokens[last_index] = "[MASK]"

    masked_lm_positions = [last_index]
    masked_lm_labels = [tokens[last_index]]

    return (output_tokens, masked_lm_positions, masked_lm_labels)

def write_instance_to_example_files(instances, max_seq_length,
                                    max_predictions_per_seq, vocab,
                                    output_files):
    """Create TF example files from `TrainingInstance`s."""
    writers = []
    for output_file in output_files:
        writers.append(tf.compat.v1.python_io.TFRecordWriter(output_file))

    writer_index = 0

    total_written = 0
    for (inst_index, instance) in enumerate(instances):
        try:
            input_ids = vocab.convert_tokens_to_ids(instance.tokens)
        except:
            print(instance)

        input_mask = [1] * len(input_ids)
        assert len(input_ids) <= max_seq_length

        input_ids += [0] * (max_seq_length - len(input_ids))
        input_mask += [0] * (max_seq_length - len(input_mask))

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length

        masked_lm_positions = list(instance.masked_lm_positions)
        masked_lm_ids = vocab.convert_tokens_to_ids(instance.masked_lm_labels)
        masked_lm_weights = [1.0] * len(masked_lm_ids)

        masked_lm_positions += [0] * (max_predictions_per_seq - len(masked_lm_positions))
        masked_lm_ids += [0] * (max_predictions_per_seq - len(masked_lm_ids))
        masked_lm_weights += [0.0] * (max_predictions_per_seq - len(masked_lm_weights))

        features = collections.OrderedDict()
        features["info"] = create_int_feature(instance.info)
        features["input_ids"] = create_int_feature(input_ids)
        features["input_mask"] = create_int_feature(input_mask)
        features["masked_lm_positions"] = create_int_feature(
            masked_lm_positions)
        features["masked_lm_ids"] = create_int_feature(masked_lm_ids)
        features["masked_lm_weights"] = create_float_feature(masked_lm_weights)

        tf_example = tf.train.Example(
            features=tf.train.Features(feature=features))

        writers[writer_index].write(tf_example.SerializeToString())
        writer_index = (writer_index + 1) % len(writers)

        total_written += 1

        if inst_index < 20:
            tf.compat.v1.logging.info("*** Example ***")
            tf.compat.v1.logging.info("tokens: %s" % " ".join(
                [printable_text(x) for x in instance.tokens]))

            for feature_name in features.keys():
                feature = features[feature_name]
                values = []
                if feature.int64_list.value:
                    values = feature.int64_list.value
                elif feature.float_list.value:
                    values = feature.float_list.value
                tf.compat.v1.logging.info("%s: %s" % (feature_name,
                                            " ".join([str(x)
                                                      for x in values])))

    for writer in writers:
        writer.close()

    tf.compat.v1.logging.info("Wrote %d total instances", total_written)

def create_int_feature(values):
    feature = tf.train.Feature(
        int64_list=tf.train.Int64List(value=list(values)))
    return feature

def create_float_feature(values):
    feature = tf.train.Feature(
        float_list=tf.train.FloatList(value=list(values)))
    return feature

def printable_text(text):
    """Returns text encoded in a way suitable for print or `tf.logging`."""

    # These functions want `str` for both Python2 and Python3, but in one case
    # it's a Unicode string and in the other it's a byte string.
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text
        elif isinstance(text, unicode):
            return text.encode("utf-8")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


class TrainingInstance(object):
    """A single training instance (sentence pair)."""

    def __init__(self, info, tokens, masked_lm_positions, masked_lm_labels):
        self.info = info  # info = [user]
        self.tokens = tokens
        self.masked_lm_positions = masked_lm_positions
        self.masked_lm_labels = masked_lm_labels

    def __str__(self):
        s = ""
        s += "info: %s\n" % (" ".join([printable_text(x) for x in self.info]))
        s += "tokens: %s\n" % (
            " ".join([printable_text(x) for x in self.tokens]))
        s += "masked_lm_positions: %s\n" % (
            " ".join([str(x) for x in self.masked_lm_positions]))
        s += "masked_lm_labels: %s\n" % (
            " ".join([printable_text(x) for x in self.masked_lm_labels]))
        s += "\n"
        return s

    def __repr__(self):
        return self.__str__()


class Data(object):
    def __init__(self, args, path):
        self.path = path
        self.args = args

        self.batch_size = args.batch_size

        train_file = path + '/train.txt'
        test_file = path + '/test.txt'

        kg_file = path + '/kg_final.txt'

        # ----------get number of users and items & then load rating data from train_file & test_file------------.
        self.n_train, self.n_test = 0, 0
        self.n_users, self.n_items = 0, 0

#[...[70677 10577][70678 17865][70678 20557]    |    {...70677:[10577], 70678: [17865, 20557]}
        self.train_data, self.train_user_dict = self._load_ratings(train_file)
        self.test_data, self.test_user_dict = self._load_ratings(test_file)
        self.exist_users = self.train_user_dict.keys()#([..., 70677, 70678])

        self._statistic_ratings()
        #得到n_train,n_test是train_data，test_data的长度。n_users，n_items是user和item的数量

#****************************************************************************************************************
        # cc = 0.0
        # max_len = 0
        # min_len = 100000
        # for u in self.train_user_dict:
        #     cc += len(self.train_user_dict[u])
        #     max_len = max(len(self.train_user_dict[u]), max_len)
        #     min_len = min(len(self.train_user_dict[u]), min_len)
        # print('average sequence length: %.2f' % (cc / len(self.train_user_dict)))
        # print('max:{}, min:{}'.format(max_len, min_len))
        #
        # b_user_train_data = {
        #     'user_' + str(u):
        #         ['item_' + str(item) for item in (self.train_user_dict[u])]
        #     for u in self.train_user_dict if len(self.train_user_dict[u]) > 0
        # }
        # b_user_test_data = {
        #     'user_' + str(u):
        #         ['item_' + str(item) for item in (self.test_user_dict[u])]
        #     for u in self.test_user_dict if len(self.test_user_dict[u]) > 0
        # }
        #
        # vocab = FreqVocab(b_user_train_data)
        # b_user_test_data_output = {  # , {..., 'user_6040': [[190, 12, 291, 63, 4, ...]]}
        #     k: [vocab.convert_tokens_to_ids(v)]
        #     for k, v in b_user_test_data.items()
        # }
        #
        # version_id = args.dataset
        # random_seed = 12345
        # short_seq_prob = 0  # Probability of creating sequences which are shorter than the maximum length。
        # rng = random.Random(random_seed)
        # max_seq_length=200
        # dupe_factor=10
        # mask_prob=1.0
        # masked_lm_prob=0.2
        # max_predictions_per_seq=40
        # prop_sliding_window=0.1
        # pool_size=10
        #
        # print('begin to generate train')
        # output_filename = path + '/' + version_id + '.train.tfrecord'
        # self.gen_samples(
        #     b_user_train_data,
        #     output_filename,
        #     rng,
        #     vocab,
        #     max_seq_length,
        #     dupe_factor,
        #     short_seq_prob,
        #     mask_prob,
        #     masked_lm_prob,
        #     max_predictions_per_seq,
        #     prop_sliding_window,
        #     pool_size,
        #     force_last=False)
        # print('train:{}'.format(output_filename))
        #
        # print('begin to generate test')
        # output_filename = path + '/' + version_id + '.test.tfrecord'
        # self.gen_samples(
        #     b_user_test_data,
        #     output_filename,
        #     rng,
        #     vocab,
        #     max_seq_length,
        #     dupe_factor,
        #     short_seq_prob,
        #     mask_prob,
        #     masked_lm_prob,
        #     max_predictions_per_seq,
        #     -1.0,
        #     pool_size,
        #     force_last=True)
        # print('test:{}'.format(output_filename))
        #
        # print('vocab_size:{}, user_size:{}, item_size:{}, item_with_other_size:{}'.
        #       format(vocab.get_vocab_size(),  # 3420
        #              vocab.get_user_count(),  # 6040
        #              vocab.get_item_count(),  # 3416  3419
        #              vocab.get_item_count() + vocab.get_special_token_count()))
        # vocab_file_name = path + '/' + version_id + '.vocab'
        # print('vocab pickle file: ' + vocab_file_name)
        # with open(vocab_file_name, 'wb') as output_file:
        #     pickle.dump(vocab, output_file, protocol=2)
        #
        # his_file_name = path + '/' + version_id + '.his'
        # print('test data pickle file: ' + his_file_name)
        # with open(his_file_name, 'wb') as output_file:
        #     pickle.dump(b_user_test_data_output, output_file, protocol=2)
        # print('done.')

# ****************************************************************************************************************

        # ----------get number of entities and relations & then load kg data from kg_file ------------.
        self.n_relations, self.n_entities, self.n_triples = 0, 0, 0
        self.kg_data, self.kg_dict, self.relation_dict = self._load_kg(kg_file)
        #kg_data: kg_final里面去重的数组
        # kg_dict:({113485: [(24916, 0), (24917, 0),...], 113486: [... (106821, 14)]}) head:[(tail,relation),(tail,relation)]
        # relation_dict:  relation:[(head,tail),(head,relation)

        # ----------print the basic info about the dataset-------------.
        self.batch_size_kg = self.n_triples // (self.n_train // self.batch_size) # //整数除法
        self._print_data_info()


    def gen_samples(self,
                    data,
                    output_filename,
                    rng,
                    vocab,
                    max_seq_length,
                    dupe_factor,
                    short_seq_prob,
                    mask_prob,
                    masked_lm_prob,
                    max_predictions_per_seq,
                    prop_sliding_window,
                    pool_size,
                    force_last=False):

        # create train
        instances = create_training_instances(
            data, max_seq_length, dupe_factor, short_seq_prob, masked_lm_prob,
            max_predictions_per_seq, rng, vocab, mask_prob, prop_sliding_window,
            pool_size, force_last)

        tf.compat.v1.logging.info("*** Writing to output files ***")
        tf.compat.v1.logging.info("  %s", output_filename)

        write_instance_to_example_files(instances, max_seq_length,
                                        max_predictions_per_seq, vocab,
                                        [output_filename])



    # reading train & test interaction data.
    def _load_ratings(self, file_name):
        user_dict = dict()
        inter_mat = list()

        lines = open(file_name, 'r').readlines()
        for l in lines:
            tmps = l.strip() #去除字符串两边的空格
            inters = [int(i) for i in tmps.split(' ')]

            u_id, pos_ids = inters[0], inters[1:]
            pos_ids = list(set(pos_ids))

            for i_id in pos_ids:
                inter_mat.append([u_id, i_id])

            if len(pos_ids) > 0:
                user_dict[u_id] = pos_ids
        return np.array(inter_mat), user_dict
        #np.array(inter_mat)：[...[70677 10577][70678 17865][70678 20557]], user_dict：{...70677:[10577], 70678: [17865, 20557]}

    def _statistic_ratings(self):
        self.n_users = max(max(self.train_data[:, 0]), max(self.test_data[:, 0])) + 1
        self.n_items = max(max(self.train_data[:, 1]), max(self.test_data[:, 1])) + 1
        self.n_train = len(self.train_data)
        self.n_test = len(self.test_data)

    # reading train & test interaction data.
    def _load_kg(self, file_name):
        def _construct_kg(kg_np):
            kg = collections.defaultdict(list)
            rd = collections.defaultdict(list) #dict_items([('yellow', [1, 3]), ('blue', [2, 4]), ('red', [1])])

            for head, relation, tail in kg_np:
                kg[head].append((tail, relation))
                rd[relation].append((head, tail))
            return kg, rd

        kg_np = np.loadtxt(file_name, dtype=np.int32)
        kg_np = np.unique(kg_np, axis=0) #去除重复行

        # self.n_relations = len(set(kg_np[:, 1]))
        # self.n_entities = len(set(kg_np[:, 0]) | set(kg_np[:, 2]))
        self.n_relations = max(kg_np[:, 1]) + 1
        self.n_entities = max(max(kg_np[:, 0]), max(kg_np[:, 2])) + 1
        self.n_triples = len(kg_np)

        kg_dict, relation_dict = _construct_kg(kg_np)

        return kg_np, kg_dict, relation_dict

    def _print_data_info(self):
        print('[n_users, n_items]=[%d, %d]' % (self.n_users, self.n_items))
        print('[n_train, n_test]=[%d, %d]' % (self.n_train, self.n_test))
        print('[n_entities, n_relations, n_triples]=[%d, %d, %d]' % (self.n_entities, self.n_relations, self.n_triples))
        print('[batch_size, batch_size_kg]=[%d, %d]' % (self.batch_size, self.batch_size_kg))

    def _generate_train_cf_batch(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.exist_users, self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]

        def sample_pos_items_for_u(u, num):
            pos_items = self.train_user_dict[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num: break
                neg_i_id = np.random.randint(low=0, high=self.n_items,size=1)[0]

                if neg_i_id not in self.train_user_dict[u] and neg_i_id not in neg_items:
                    neg_items.append(neg_i_id)
            return neg_items

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)

        return users, pos_items, neg_items

    def get_sparsity_split(self):
        try:
            split_uids, split_state = [], []
            lines = open(self.path + '/sparsity.split', 'r').readlines()

            for idx, line in enumerate(lines):
                if idx % 2 == 0:
                    split_state.append(line.strip())
                    print(line.strip())
                else:
                    split_uids.append([int(uid) for uid in line.strip().split(' ')])
            print('get sparsity split.')

        except Exception:
            split_uids, split_state = self.create_sparsity_split()
            f = open(self.path + '/sparsity.split', 'w')
            for idx in range(len(split_state)):
                f.write(split_state[idx] + '\n')
                f.write(' '.join([str(uid) for uid in split_uids[idx]]) + '\n')
            print('create sparsity split.')

        return split_uids, split_state



    def create_sparsity_split(self):
        all_users_to_test = list(self.test_user_dict.keys())
        user_n_iid = dict()

        # generate a dictionary to store (key=n_iids, value=a list of uid).
        for uid in all_users_to_test:
            train_iids = self.train_user_dict[uid]
            test_iids = self.test_user_dict[uid]

            n_iids = len(train_iids) + len(test_iids)

            if n_iids not in user_n_iid.keys():
                user_n_iid[n_iids] = [uid]
            else:
                user_n_iid[n_iids].append(uid)
        split_uids = list()

        # split the whole user set into four subset.
        temp = []
        count = 1
        fold = 4
        n_count = (self.n_train + self.n_test)
        n_rates = 0

        split_state = []
        for idx, n_iids in enumerate(sorted(user_n_iid)):
            temp += user_n_iid[n_iids]
            n_rates += n_iids * len(user_n_iid[n_iids])
            n_count -= n_iids * len(user_n_iid[n_iids])

            if n_rates >= count * 0.25 * (self.n_train + self.n_test):
                split_uids.append(temp)

                state = '#inter per user<=[%d], #users=[%d], #all rates=[%d]' %(n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)

                temp = []
                n_rates = 0
                fold -= 1

            if idx == len(user_n_iid.keys()) - 1 or n_count == 0:
                split_uids.append(temp)

                state = '#inter per user<=[%d], #users=[%d], #all rates=[%d]' % (n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)


        return split_uids, split_state