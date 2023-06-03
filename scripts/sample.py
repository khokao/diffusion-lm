import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', type=str, help='Directory where the training results are saved')
    parser.add_argument('-n', '--n_samples', type=int, default=16, help='Number to be sampled (used as batch size)')
    parser.add_argument('-mc', '--model_ckpt', type=str, default='checkpoints/pytorch_model_1.bin',
                        help='Path to Diffusion-LM checkpoint (from the path specified in the `output` argument)')
    parser.add_argument('-ud', '--use_ddpm', action='store_true', default=False,
                        help='Whether to use DDPM sample (default is DDIM)')
    parser.add_argument('-cc', '--clf_ckpt', type=str, default='classifier/pytorch_model.bin',
                        help='Path to classifier checkpoint (from the path specified in the `output` argument)')
    parser.add_argument('-cl', '--control_label', type=str, default=None, help='Label for pnp control')
    args = parser.parse_args()
    return args


def main():
    args = vars(get_args())

    module = __import__('difflm', fromlist=['DiffusionLMInterface'])
    interface_cls = getattr(module, 'DiffusionLMInterface')
    interface = interface_cls(args, mode='infer')

    interface.sample(n_samples=args['n_samples'], use_ddpm=args['use_ddpm'], control_label=args['control_label'])


if __name__ == '__main__':
    main()
