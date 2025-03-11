import os
import torch
import argparse
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_collection import AudioSet
from utilities import collate_func, set_logger, write_json, collate_func
import sys

sys.path.append("../src/lavis")
from lavis.models import load_model_and_preprocess

log = set_logger(__name__)


def generate_answer_on_audioset(
    lam_ckpt_path="",
    results_json_path="/data/home/eey340/WORKPLACE/LAM/engine/results/test_fsl.json",
    n_ways: int = 4,
    mini_data=True,
):
    sys.path.append("/data/home/eey340/WORKPLACE/LAM/src/lavis/lavis/datasets/datasets")
    from few_shot_datasets import AudioSetFSL

    dataset = AudioSetFSL(
        vis_processor=lambda x: x,
        audio_root="/data/EECS-MachineListeningLab/datasets/AudioSet/audios",
        seg_name="eval",
    )
    # Outline the given audio samples briefly. Summarize the audio with key words.
    # prompt = "Summarize the audio with key words"
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    # loads lam pre-trained model
    model, _, _ = load_model_and_preprocess(
        name="lam_vicuna_instruct",
        model_type="vicuna7b-ft",  # vicuna1.5_7b-ft
        is_eval=True,
        device=device,
    )

    model.load_from_pretrained(lam_ckpt_path)

    # Load sample audio
    # evaluate one episode each batch
    dataset = AudioSetFSL(
        vis_processor=lambda x: x,
        audio_root="/data/EECS-MachineListeningLab/datasets/AudioSet/audios",
        seg_name="eval",
    )

    results = []
    with tqdm(total=10 if mini_data else len(dataset)) as pbar:
        for idx, data in enumerate(dataset):
            # NOTE: change to gererate_new if bos is in the first position.
            # Before 9/1 use `generate` method
            # prompt = []
            # for id, p in enumerate(data[3]):
            #     if id == len(data[3]) - 1:
            #         prompt.append("This is a sound of ")
            #         ground_truth = p
            #     else:
            #         prompt.append("This is a sound of " + "; ".join(p))

            prompt = [
                "This is a sound of Class A",
                "This is a sound of Class B",
                "This is a sound of Class C",
                "This is a sound of Class D",
                "This is a sound of",
            ]

            output = model.generate(
                {
                    "audio": data["audio"].cuda().unsqueeze(0),
                    "prompt": prompt,  # data["text_input"],
                },
                temperature=0.1,
            )

            print(output)
            print(data["text_output"])
            print(data["text_input"])
            # # output = torch.tensor_split(output, batch_size, dim=0)
            # for fname, predicted_answer, tgt_cat, tgt_multihot in zip(
            #         data[0], output, data[3], data[6]):
            #     res = {
            #         "file_name":
            #         fname,
            #         "predicted_answer":
            #         predicted_answer,
            #         "target_label":
            #         tgt_cat,
            #         "ground_truth":
            #         ",".join(
            #             [str(int(elem)) for elem in tgt_multihot.tolist()]),
            #     }
            #     log.info(res)
            #     results.append(res)

            # pbar.update(1)
            if mini_data and idx == 10:
                break

    # write_json(results, results_json_path)


def _get_category_tokens(
    swift,
    csv_path="/data/EECS-MachineListeningLab/datasets/AudioSet/meta/class_labels_indices.csv",
):
    r"""Return text embeddings of audioSet labels using swift encoder, as well as max, min values of similarities across labels."""
    import pandas as pd

    df = pd.read_csv(csv_path)
    categories_tokens = []
    for _, r in df.iterrows():
        cat = r["display_name"].lower()
        # Use the everage of word embeddings as text embedding
        text_emb = swift.get_text_embedding(cat, padding=False)["text_embedding"].mean(
            dim=1
        )
        # Use the class embedding as text embeddings
        # text_emb = swift.get_text_embedding(cat, padding=False)["class_embedding"]

        categories_tokens.append(text_emb)

    categories_tokens = torch.cat(categories_tokens)

    # Get the max and min values of similarties between categories
    sim = swift.compute_similarity(categories_tokens, categories_tokens)
    sim_min, sim_max = sim.min(), sim.max()

    return categories_tokens, sim_min, sim_max


def _min_max_normalization(x, ori_min, ori_max, tgt_min, tgt_max):
    res = (x - ori_min) / (ori_max - ori_min)
    return (tgt_max - tgt_min) * res + tgt_min


def evaluate_generated_answer(
    results_json_path,
    swift_ckpt_path,
    normalize_sim=False,
    mini_data=True,
):
    from torchmetrics import MetricCollection
    from torchmetrics.classification import (
        MultilabelAccuracy,
        MultilabelF1Score,
        MultilabelAveragePrecision,
        MultilabelAUROC,
    )
    from utilities import read_json

    # sys.path.append("../src/lavis")
    # from lavis.models.lam_models.swift import SWIFT
    from models.swift import SWIFT

    num_class = 527
    # temp = torch.nn.Parameter(0.07 * torch.ones([]))  # 0.07
    metric = MetricCollection(
        [
            MultilabelAccuracy(num_labels=num_class, average="micro", threshold=0.002),
            MultilabelF1Score(num_labels=num_class, average="macro", threshold=0.002),
            MultilabelAveragePrecision(num_labels=num_class, average="macro"),
            MultilabelAUROC(num_labels=num_class, average="macro"),
        ]
    )
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    swift = SWIFT()
    swift.init_swift(device=device)
    swift.load_checkpoint(swift_ckpt_path)
    swift = swift.to(device)

    category_tokens, sim_min, sim_max = _get_category_tokens(swift)

    data = read_json(results_json_path)

    with tqdm(total=10 if mini_data else len(data)) as pbar:
        for idx, datum in enumerate(data):
            text_emb = swift.get_text_embedding(
                datum["predicted_answer"],
                padding=False,
            )["text_embedding"].squeeze(0)

            try:
                sim, _ = swift.compute_similarity(text_emb, category_tokens).max(dim=0)
            except:
                log.warning("Empty generated answer. Use `zero` instead.")
                sim = torch.zeros(num_class)

            if normalize_sim:
                sim = _min_max_normalization(
                    sim,
                    ori_min=sim_min,
                    ori_max=sim_max,
                    tgt_min=-1,
                    tgt_max=1,
                )
            # sim /= temp
            sim = sim.sigmoid()

            ground_truth = [int(elem) for elem in datum["ground_truth"].split(",")]
            ground_truth = torch.tensor(ground_truth).unsqueeze(0).long()

            metric.update(sim.unsqueeze(0).cpu(), ground_truth)

            pbar.update(1)
            if mini_data and idx == 9:
                break

    result = metric.compute()
    log.info(f"Save the results to {results_json_path}")
    log.info(result)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_folder_name", type=str)
    args = parser.parse_args()

    lam_ckpt_dir = (
        "/data/EECS-MachineListeningLab/jinhua/lam/check_point/Pretrain_stage2"
    )
    # "/data/EECS-MachineListeningLab/jinhua/lam/check_point/Pretrain_stage2_audioset_aqa/20230814203/checkpoint_30000.pth"
    results_json_dir = "/data/home/eey340/WORKPLACE/LAM/engine/results"
    swift_ckpt_path = "/data/EECS-MachineListeningLab/jinhua/lam/check_point/Pretrain_stage1/20230825051/checkpoint_14700.pth"
    mini_data = False
    normalize_sim = False

    ckpt_foldername = args.ckpt_folder_name  # "20230829124"
    # for pth in [
    #         "checkpoint_62500.pth",
    #         "checkpoint_125000.pth",
    #         "checkpoint_250000.pth",
    #         "checkpoint_375000.pth",
    #         "checkpoint_500000.pth",
    #         "checkpoint_625000.pth",
    #         "checkpoint_687500.pth",
    #         "checkpoint_187500.pth",
    #         "checkpoint_312500.pth",
    #         "checkpoint_437500.pth",
    #         "checkpoint_562500.pth",
    #         "checkpoint_750000.pth",
    # ]:
    # for pth in [
    #         "checkpoint_62500.pth",
    #         "checkpoint_125000.pth",
    #         "checkpoint_312500.pth",
    #         "checkpoint_187500.pth",
    #         "checkpoint_375000.pth",
    #         "checkpoint_562500.pth",
    #         "checkpoint_687500.pth",
    #         "checkpoint_875000.pth",
    #         "checkpoint_250000.pth",
    #         "checkpoint_437500.pth",
    #         "checkpoint_625000.pth",
    #         "checkpoint_750000.pth",
    #         "checkpoint_937500.pth",
    #         "checkpoint_500000.pth",
    #         "checkpoint_812500.pth",
    #         "checkpoint_1000000.pth",
    #         "checkpoint_1062500.pth",
    #         "checkpoint_1125000.pth",
    #         "checkpoint_1187500.pth",
    #         "checkpoint_1250000.pth",
    # ]:
    for pth in [
        # "checkpoint_250000.pth",
        # "checkpoint_500000.pth",
        # "checkpoint_750000.pth",
        # "checkpoint_1000000.pth",
        # "checkpoint_1250000.pth",
        # "checkpoint_1500000.pth",
        # "checkpoint_1750000.pth",
        "checkpoint_2000000.pth",
    ]:
        lam_ckpt_path = os.path.join(lam_ckpt_dir, ckpt_foldername, pth)
        results_json_path = os.path.join(
            results_json_dir, f"{ckpt_foldername}_{pth}.json"
        )
        generate_answer_on_audioset(
            lam_ckpt_path=lam_ckpt_path,
            results_json_path=results_json_path,
            mini_data=mini_data,
        )
        # evaluate_generated_answer(
        #     results_json_path=results_json_path,
        #     swift_ckpt_path=swift_ckpt_path,
        #     normalize_sim=normalize_sim,
        #     mini_data=mini_data,
        # )
