import argparse
from accelerate import notebook_launcher


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', type=str, help='Directory where the training results are saved')
    parser.add_argument('-mc', '--model_ckpt', type=str, default='checkpoints/pytorch_model_1.bin',
                        help='Path to Diffusion-LM checkpoint (from the path specified in the `output` argument)')
    args = parser.parse_args()
    return args


def main():
    args = vars(get_args())

    module = __import__('difflm', fromlist=['DiffusionLMInterface'])
    interface_cls = getattr(module, 'DiffusionLMInterface')
    interface = interface_cls(args, mode='clf_train')

    notebook_launcher(
        interface.clf_train,
        args=(),
        num_processes=1,
    )


if __name__ == '__main__':
    main()
