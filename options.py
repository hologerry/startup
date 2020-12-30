import argparse


def get_base_parser():
    parser = argparse.ArgumentParser()
    # Azure or local
    parser.add_argument('--azure', type=bool, default=True, help='Azure or not determines the workspace directory')
    parser.add_argument('--multi_nodes', type=bool, default=True, help='Use multiple machines with DDP')
    parser.add_argument("--local_rank", type=int, help="Local rank. Necessary for using the torch.distributed.launch utility.")
    # Experiment
    parser.add_argument('--exper_root', type=str, default='experiments', help='experiments root dir')
    # Model
    parser.add_argument('--exper_name', type=str,
                        default='exper_name',
                        help='experiment name, to use different model, modify the import xx as GAN in train.py or test.py')
    # Data
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--dataset_name', type=str, default='dataset_name')
    parser.add_argument('--image_size', type=str, default='image_size')
    return parser


def get_train_parser():
    parser = get_base_parser()
    parser.add_argument('--mode', type=str, default='train')
    # Experiment
    parser.add_argument('--init_epoch', type=int, default=0, help='epoch to start training from')
    parser.add_argument('--epochs', type=int, default=200, help='number of training epochs')
    parser.add_argument('--fine_tune', type=bool, default=False, help='fine tune the whole model')
    parser.add_argument('--model_resume', type=str, default='model_resume', help='path to load the pretrained baseline model')
    # Data
    parser.add_argument('--batch_size', type=int, default=1)
    # Optimizer
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--beta1', type=float, default=0.0, help='beta1 of adam')
    parser.add_argument('--beta2', type=float, default=0.99, help='beta2 of adam')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum of sgd')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay of adam')
    parser.add_argument('--decay_epochs', type=int, default=50, help='after x epochs, decay lr')
    parser.add_argument('--decay_step', type=int, default=5, help='decay lr every x steps')
    parser.add_argument('--decay_ratio', type=float, default=0.999, help='decay lr ratio')

    # Lambda
    parser.add_argument('--lambda_', type=float, default=0.1, help='lambda of adversarial')
    # Frequency
    parser.add_argument('--ckpt_freq', type=int, default=2, help='save checkpoint frequency of epochs')
    parser.add_argument('--sample_freq', type=int, default=1000, help='sample frequency of iterations')
    parser.add_argument('--log_freq', type=int, default=500, help='save training logs')
    return parser


def get_test_parser():
    parser = get_base_parser()
    parser.add_argument('--mode', type=str, default='test', help='test ')
    # Experiment
    parser.add_argument('--test_epoch', type=int, default=0, help='epoch to test')
    # Data
    parser.add_argument('--batch_size', type=int, default=10)
    return parser


def get_evaluate_parser():
    parser = get_base_parser()
    parser.add_argument('--mode', type=str, default='eval',
                        help='evaluation mode')
    # Experiment
    parser.add_argument('--test_epoch', type=int, default=100, help='epoch to evaluate')
    # Data
    parser.add_argument('--batch_size', type=int, default=10)

    return parser
