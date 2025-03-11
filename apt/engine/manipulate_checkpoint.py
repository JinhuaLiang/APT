import torch
from torch import Tensor
from typing import Union


def trim_lam_checkpoint(
    input_ckpt_path,
    output_ckpt_path,
    tgt_module_name="llm_model",
):
    r"""This is originally for trimming audio-language models by replacing the frozen language model."""
    from copy import deepcopy
    from tqdm import tqdm

    state_dict = torch.load(input_ckpt_path, map_location=torch.device("cpu"))

    check_point = deepcopy(state_dict["model"])
    with tqdm(total=len(check_point)) as pbar:
        for k, v in check_point.items():
            if tgt_module_name in k:
                del state_dict["model"][k]
            pbar.update(1)

    torch.save(state_dict, output_ckpt_path)


def average_models(models: list, ratios: Union[list, Tensor, None] = None):
    def weighted_sum(tensors, weights):
        result = torch.zeros_like(tensors[0], dtype=torch.float32)
        for w, t in zip(ratios, tensors):
            result += w * t
        return result

    if ratios == None:
        ratios = [1 / len(models)] * len(models)

    assert len(models) == len(
        ratios
    ), f"length of `models` should identical to length of `ratios`."

    total_ratio = sum(ratios)
    if total_ratio != 1:
        _tmp = [r / total_ratio for r in ratios]
        ratios = _tmp

    template_model = models[0]
    for idx, mdl in enumerate(models):
        assert set(template_model.keys()) == set(
            mdl.keys()
        ), f"item {idx} in models is not in the same structure."

    return {
        key: weighted_sum([mdl[key] for mdl in models], ratios)
        for key in template_model.keys()
    }


if __name__ == "__main__":
    r"""Test."""
    # model_0 = {"foo": torch.tensor([5, 5, 5]).float()}
    # model_1 = {"foo": torch.tensor([4, 4, 4]).float()}

    # ratios = [2, 5]
    # result = average_models([model_0, model_1], ratios)
    # print(result)
    r"""Trim models"""
    # input_ckpt_path = "/data/EECS-MachineListeningLab/jinhua/lam/check_point/Pretrain_stage2_audioset_aqa/20230814203/checkpoint_90000.pth"
    # output_ckpt_path = input_ckpt_path
    # # output_ckpt_path = "./20230824102_625000.pth"
    # trim_lam_checkpoint(input_ckpt_path, output_ckpt_path)
    # print(torch.load(output_ckpt_path).keys())
    # print(torch.load(output_ckpt_path)['model'].keys())
    # This is for batch trim the weights under the same folder
    import os
    from glob import glob

    input_ckpt_dir = "/data/EECS-MachineListeningLab/jinhua/lam/check_point/Pretrain_stage2/20230826222"
    checkpoint_names = glob(os.path.join(input_ckpt_dir, "*.pth"))
    for input_ckpt_path in checkpoint_names:
        # for ckpt_name in [
        #     "checkpoint_62500.pth",
        #     "checkpoint_125000.pth",
        #     "checkpoint_187500.pth",
        #     "checkpoint_250000.pth",
        #     "checkpoint_312500.pth",
        #     "checkpoint_375000.pth",
        #     "checkpoint_437500.pth",
        #     "checkpoint_500000.pth",
        #     "checkpoint_562500.pth",
        #     "checkpoint_625000.pth",
        #     "checkpoint_687500.pth",
        #     "checkpoint_750000.pth",
        #     "checkpoint_812500.pth",
        #     "checkpoint_875000.pth",
        #     "checkpoint_937500.pth",
        #     "checkpoint_1000000.pth",
        #     "checkpoint_1062500.pth",
        #     "checkpoint_1125000.pth",
        #     "checkpoint_1187500.pth",
        #     "checkpoint_1250000.pth",
        # ]:
        # input_ckpt_path = os.path.join(input_ckpt_dir, ckpt_name)
        output_ckpt_path = input_ckpt_path
        trim_lam_checkpoint(input_ckpt_path, output_ckpt_path)
    r"""Doing the model soup."""
    # model_0 = torch.load('./20230824102_625000.pth')['model']
    # model_1 = torch.load('./20230826222_625000.pth')['model']

    # result_model = average_models(models=[model_0, model_1], ratios=[1.0, 1.0])
    # torch.save({"model": result_model}, "./result_model.pth")
