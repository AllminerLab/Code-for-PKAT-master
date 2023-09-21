from collections import Counter


def convert_by_vocab(vocab, items):
    """Converts a sequence of [tokens|ids] using the vocab."""
    output = []
    for item in items:
        output.append(vocab[item])
    return output


class FreqVocab(object):
    """Runs end-to-end tokenziation."""

    def __init__(self, user_to_list):
        # layout of the  ulary
        # item_id based on freq
        # special token
        # user_id based on nothing
        self.counter = Counter(
        )  #sorted(self.items(), key=_itemgetter(1), reverse=True)
        self.user_set = set()
        for u, item_list in user_to_list.items():
            self.counter.update(item_list)
            self.user_set.add(str(u))
        #Counter({'item_10': 3406, 'item_150': 2981, ...})
        self.user_count = len(self.user_set)
        self.item_count = len(self.counter.keys())
        self.special_tokens = {"[pad]", "[MASK]", '[NO_USE]'}
        self.token_to_ids = {}  # index begin from 1
        #first items {'item_10': 1, 'item_150': 2, 'item_145': 3...'item_3252': 3416}
        for token, count in self.counter.most_common():
            self.token_to_ids[token] = len(self.token_to_ids) + 1

        # then special tokens {'item_10': 1, 'item_150': 2,...,'[NO_USE]': 3417, '[MASK]': 3418, '[pad]': 3419}
        for token in self.special_tokens:
            self.token_to_ids[token] = len(self.token_to_ids) + 1

        # then user
#         for user in self.user_set:
#             self.token_to_ids[user] = len(self.token_to_ids) + 1

        self.id_to_tokens = {v: k for k, v in self.token_to_ids.items()} #{1: 'item_10', 2: 'item_150', 3: 'item_145',...}
        self.vocab_words = list(self.token_to_ids.keys()) #['item_10', 'item_150', 'item_145',...]

    def convert_tokens_to_ids(self, tokens):
        return convert_by_vocab(self.token_to_ids, tokens)

    def convert_ids_to_tokens(self, ids):
        return convert_by_vocab(self.id_to_tokens, ids)

    def get_vocab_words(self):
        return self.vocab_words  # not in order

    def get_item_count(self):
        return self.item_count

    def get_user_count(self):
        return self.user_count

    def get_items(self):
        return list(self.counter.keys())

    def get_users(self):
        return self.user_set

    def get_special_token_count(self):
        return len(self.special_tokens)

    def get_special_token(self):
        return self.special_tokens

    def get_vocab_size(self):
        return self.get_item_count() + self.get_special_token_count() + 1 #self.get_user_count()
