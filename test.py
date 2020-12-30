import os
from os.path import join as ospj

from tqdm import tqdm

from options import get_test_parser
from data_loader import get_dataloader

from models.simple_model import Model


def test(opts):
    print(f"Testing {opts.mode} on experiment {opts.exper_name}...")

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    opts.multi_nodes = False

    exper_dir = ospj(opts.exper_root, opts.exper_name)
    ckpt_dir = ospj(exper_dir, "checkpoint")
    result_dir = ospj(exper_dir, "results")

    os.makedirs(result_dir, exist_ok=True)

    test_dataloader = get_dataloader(opts.data_root, opts.dataset_name, opts.case_name, opts.sketch_mode, 'test', opts.multi_nodes,
                                     opts.one_image_times, opts.size_range, opts.style_size, opts.batch_size)
    gan = Model(opts)
    model_file_path = ospj(ckpt_dir, f"model_{opts.test_epoch}.pth")
    gan.load_ckpt(model_file_path)
    gan.eval()
    gan.parallel()

    for test_batch_idx, test_data_batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        save_file = ospj(result_dir, f"test_epoch_{opts.test_epoch}_{opts.dataset_name}_batch_{test_batch_idx}.png")
        gan.predict(test_data_batch, save_file)


if __name__ == "__main__":
    parser = get_test_parser()
    opts = parser.parse_args()
    test(opts)
