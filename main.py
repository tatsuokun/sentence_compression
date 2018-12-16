import argparse
import torch
from run import load, run


def parse_args():

    gpu_id = -1
    parser = argparse.ArgumentParser(prog='evaluation')
    parser.add_argument('--gpu-id',
                        type=int,
                        metavar='GPU_ID',
                        default=gpu_id)
    return parser.parse_args()


def main():

    args = parse_args()
    gpu_id = args.gpu_id
    use_cuda = torch.cuda.is_available() and gpu_id > -1
    if use_cuda:
        torch.cuda.set_device(gpu_id)
    batch_size = 50
    n_epochs = 5
    train, dev, test = load('data/google_com_train.conll',
                            'data/google_com_dev.conll',
                            'data/google_com_test.conll',
                            batch_size=batch_size,
                            device=gpu_id)
    run(train,
        dev,
        test,
        model_type='base',
        word_embed_size=50,
        hidden_size=256,
        batch_size=50,
        use_cuda=use_cuda,
        n_epochs=n_epochs)


if __name__ == '__main__':
    main()
