import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--expn', type=str, help='Experiment name (basename of output directory)', default=None)
    parser.add_argument('-w', '--wandb', action='store_true', help='Whether to use wandb')
    args = parser.parse_args()
    return args


def main():
    args = vars(get_args())

    module = __import__('difflm', fromlist=['DiffusionLMInterface'])
    interface_cls = getattr(module, 'DiffusionLMInterface')
    interface = interface_cls(args, mode='train')

    interface.train()


if __name__ == '__main__':
    main()
