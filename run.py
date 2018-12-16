import torch
from torch import optim
from const import Phase
from batch import create_dataset
from models import Baseline
from sklearn.metrics import classification_report


def run(dataset_train,
        dataset_dev,
        dataset_test,
        model_type,
        word_embed_size,
        hidden_size,
        batch_size,
        use_cuda,
        n_epochs):

    if model_type == 'base':
        model = Baseline(vocab=dataset_train.vocab,
                         word_embed_size=word_embed_size,
                         hidden_size=hidden_size,
                         use_cuda=use_cuda,
                         inference=False)
    else:
        raise NotImplementedError
    if use_cuda:
        model = model.cuda()

    optim_params = model.parameters()
    optimizer = optim.Adam(optim_params, lr=10**-3)

    print('start training')
    for epoch in range(n_epochs):
        train_loss, tokens, preds, golds = train(dataset_train,
                                                 model,
                                                 optimizer,
                                                 batch_size,
                                                 epoch,
                                                 Phase.TRAIN,
                                                 use_cuda)

        dev_loss, tokens, preds, golds = train(dataset_dev,
                                               model,
                                               optimizer,
                                               batch_size,
                                               epoch,
                                               Phase.DEV,
                                               use_cuda)
        logger = '\t'.join(['epoch {}'.format(epoch+1),
                            'TRAIN Loss: {:.9f}'.format(train_loss),
                            'DEV Loss: {:.9f}'.format(dev_loss)])
        print('\r'+logger, end='')
    test_loss, tokens, preds, golds = train(dataset_test,
                                            model,
                                            optimizer,
                                            batch_size,
                                            epoch,
                                            Phase.TEST,
                                            use_cuda)
    print('====', 'TEST', '=====')
    print_scores(preds, golds)
    output_results(tokens, preds, golds)


def train(dataset,
          model,
          optimizer,
          batch_size,
          n_epoch,
          phase,
          use_cuda):

    total_loss = 0.0
    tokens = []
    preds = []
    labels = []
    if phase == Phase.TRAIN:
        model.train()
    else:
        model.eval()

    for batch in dataset.batch_iter:
        token = getattr(batch, 'token')
        label = getattr(batch, 'label')
        if use_cuda:
            raw_sentences = dataset.get_raw_sentence(token.data.cpu().numpy())
        else:
            raw_sentences = dataset.get_raw_sentence(token.data.numpy())

        loss, pred = \
            model(token, raw_sentences, label, phase)

        if phase == Phase.TRAIN:
            optimizer.zero_grad()
            torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=5)
            loss.backward()
            optimizer.step()

        # remove PAD from input sentences/labels and results
        mask = (token != dataset.pad_index)
        length_tensor = mask.sum(1)
        if use_cuda:
            length_tensor = length_tensor.data.cpu().numpy()
        else:
            length_tensor = length_tensor.data.numpy()

        for index, n_tokens_in_the_sentence in enumerate(length_tensor):
            if n_tokens_in_the_sentence > 0:
                tokens.append(raw_sentences[index][:n_tokens_in_the_sentence])
                _label = label[index][:n_tokens_in_the_sentence]
                _pred = pred[index][:n_tokens_in_the_sentence]
                if use_cuda:
                    _label = _label.data.cpu().numpy()
                    _pred = _pred.data.cpu().numpy()
                else:
                    _label = _label.data.numpy()
                    _pred = _pred.data.numpy()
                labels.append(_label)
                preds.append(_pred)

        total_loss += loss.data.mean()

    return total_loss, tokens, preds, labels


def read_two_cols_data(fname):
    data = {}
    tokens = []
    labels = []
    token = []
    label = []
    with open(fname, mode='r') as f:
        for line in f:
            line = line.strip().lower().split()
            if line:
                try:
                    _token, _label = line
                except ValueError:
                    raise
                token.append(_token)
                if _label == '0' or _label == '1':
                    label.append(int(_label))
                else:
                    if _label == 'del':
                        label.append(1)
                    else:
                        label.append(0)
            else:
                tokens.append(token)
                labels.append(label)
                token = []
                label = []

    data['tokens'] = tokens
    data['labels'] = labels
    return data


def load(train_path, dev_path, test_path, batch_size, device):
    train = read_two_cols_data(train_path)
    dev = read_two_cols_data(dev_path)
    test = read_two_cols_data(test_path)
    data = {Phase.TRAIN: train, Phase.DEV: dev, Phase.TEST: test}
    return create_dataset(data, batch_size=batch_size, device=device)


def print_scores(preds, golds):
    _preds = [label for sublist in preds for label in sublist]
    _golds = [label for sublist in golds for label in sublist]
    target_names = ['not_del', 'del']
    print(classification_report(_golds, _preds, target_names=target_names, digits=5))


def output_results(tokens, preds, golds, path='./result/sentcomp'):
    with open(path+'.original.txt', mode='w') as w, \
            open(path+'.gold.txt', mode='w') as w_gold, \
            open(path+'.pred.txt', mode='w') as w_pred:

        for _tokens, _golds, _preds in zip(tokens, golds, preds):
            for token, gold, pred in zip(_tokens, _golds, _preds):
                w.write(token + ' ')
                if gold == 0:
                    w_gold.write(token + ' ')
                # 0 -> keep, 1 -> delete
                if pred == 0:
                    w_pred.write(token + ' ')
            w.write('\n')
            w_gold.write('\n')
            w_pred.write('\n')
