from torchtext import data
from const import Phase


def create_dataset(data: dict, batch_size: int, device: int):

    train = Dataset(data[Phase.TRAIN]['tokens'],
                    data[Phase.TRAIN]['labels'],
                    vocab=None,
                    batch_size=batch_size,
                    device=device,
                    phase=Phase.TRAIN)

    dev = Dataset(data[Phase.DEV]['tokens'],
                  data[Phase.DEV]['labels'],
                  vocab=train.vocab,
                  batch_size=batch_size,
                  device=device,
                  phase=Phase.DEV)

    test = Dataset(data[Phase.TEST]['tokens'],
                   data[Phase.TEST]['labels'],
                   vocab=train.vocab,
                   batch_size=batch_size,
                   device=device,
                   phase=Phase.TEST)
    return train, dev, test


class Dataset:
    def __init__(self,
                 tokens: list,
                 label_list: list,
                 vocab: list,
                 batch_size: int,
                 device: int,
                 phase: Phase):
        assert len(tokens) == len(label_list), \
            'the number of sentences and the number of POS/head sequences \
             should be the same length'

        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'
        self.tokens = tokens
        self.label_list = label_list
        self.sentence_id = [[i] for i in range(len(tokens))]
        self.device = device

        self.token_field = data.Field(use_vocab=True,
                                      unk_token=self.unk_token,
                                      pad_token=self.pad_token,
                                      batch_first=True)
        self.label_field = data.Field(use_vocab=False, pad_token=-1, batch_first=True)
        self.sentence_id_field = data.Field(use_vocab=False, batch_first=True)
        self.dataset = self._create_dataset()

        if vocab is None:
            self.token_field.build_vocab(self.tokens)
            self.vocab = self.token_field.vocab
        else:
            self.token_field.vocab = vocab
            self.vocab = vocab
        self.pad_index = self.token_field.vocab.stoi[self.pad_token]

        self._set_batch_iter(batch_size, phase)

    def get_raw_sentence(self, sentences):
        return [[self.vocab.itos[idx] for idx in sentence]
                for sentence in sentences]

    def _create_dataset(self):
        _fields = [('token', self.token_field),
                   ('label', self.label_field),
                   ('sentence_id', self.sentence_id_field)]
        return data.Dataset(self._get_examples(_fields), _fields)

    def _get_examples(self, fields: list):
        ex = []
        for sentence, label, sentence_id in zip(self.tokens, self.label_list, self.sentence_id):
            ex.append(data.Example.fromlist([sentence, label, sentence_id], fields))
        return ex

    def _set_batch_iter(self, batch_size: int, phase: Phase):

        def sort(data: data.Dataset) -> int:
            return len(getattr(data, 'token'))

        train = True if phase == Phase.TRAIN else False

        self.batch_iter = data.BucketIterator(dataset=self.dataset,
                                              batch_size=batch_size,
                                              sort_key=sort,
                                              train=train,
                                              repeat=False,
                                              device=self.device)
