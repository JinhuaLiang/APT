import torch
import argparse
from PIL import Image
from torch.utils.data import DataLoader

from utilities import set_logger, write_json, collate_func
import sys

# log = set_logger(__name__)

# def parse_args():
#     parser = argparse.ArgumentParser(
#         description='Easy audio feature extractor')

#     parser.add_argument(
#         "--csv_dir",
#         type=str,
#         help=
#         "Should be a csv file of a single columns, each row is the input video path."
#     )
#     parser.add_argument("--audio_output_dir",
#                         type=str,
#                         help="The place to store the video frames.")
#     parser.add_argument("--mini_data",
#                         action='store_true',
#                         default=False,
#                         help="Test on mini_batch.")
#     args = parser.parse_args()

#     return args


def generate_answer_on_audioset():
    from data_collection import AudioSet
    from utilities import collate_func

    sys.path.append("../src/lavis")
    from lavis.models import load_model_and_preprocess

    # setup device to use
    batch_size = 10
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    # loads BLIP-2 pre-trained model
    model, _, _ = load_model_and_preprocess(
        name="lam_vicuna_instruct",
        model_type="vicuna7b",
        is_eval=True,
        device=device,
    )

    # load sample image
    dataset = AudioSet.build_dataset({
        "dataset_name": "eval",
    })
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_func,
    )

    results = []
    for data in loader:
        output = model.generate({
            "audio":
            data[2].cuda(),
            "prompt":
            "Describe the sound event in the audio clip.",
        })

        # output = torch.tensor_split(output, batch_size, dim=0)
        for fname, predicted_answer, tgt_cat, tgt_multihot in zip(
                data[0], output, data[3], data[6]):
            res = {
                "file_name":
                fname,
                "predicted_answer":
                predicted_answer,
                "target_label":
                tgt_cat,
                "ground_truth":
                ",".join([str(elem) for elem in tgt_multihot.tolist()]),
            }
            print(res)
            # results.append(res)
            break

    # write_json(
    #     results,
    #     "/data/home/eey340/WORKPLACE/LAM/engine/results/lam_on_audioset_val_new2.json"
    # )


def get_category_tokens(swift):
    import pandas as pd
    from utilities import dump_pickle

    sys.path.append("../src/lavis")
    from lavis.models.lam_models.swift import SWIFT

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    df = pd.read_csv(
        "/data/EECS-MachineListeningLab/datasets/AudioSet/meta/class_labels_indices.csv"
    )
    categories = [r["display_name"].lower() for _, r in df.iterrows()]

    # swift = SWIFT()
    # swift.init_swift(device=device)
    # swift.load_checkpoint(
    #     # "/data/EECS-MachineListeningLab/jinhua/lam/check_point/Pretrain_stage2_audioset_aqa/20230803203/checkpoint_90000.pth"
    #     ckpt_path)

    # categories_tokens = {
    #     cat: swift.get_text_embedding(cat)
    #     for cat in categories
    # }

    # print(categories_tokens)
    # dump_pickle(categories_tokens, "categories.pickle")
    categories_tokens, cats = [], []
    for id in range(len(categories)):
        cat = categories[id]
        cats.append(cat)
        # emb = swift.get_text_embedding(cat, padding=False)["class_embedding"]
        emb = swift.get_text_embedding(
            cat, padding=False)["text_embedding"].mean(dim=1)

        categories_tokens.append(emb)
        # break

    categories_tokens = torch.cat(categories_tokens, dim=0)
    print(categories_tokens)
    # dump_pickle(categories_tokens, "./categories.pickle")
    return categories_tokens


def evaluate_on_audioset(ckpt_path):
    from torchmetrics import MetricCollection
    from torchmetrics.classification import (
        MulticlassAccuracy, MulticlassF1Score, MultilabelAccuracy,
        MultilabelF1Score, MultilabelAveragePrecision, MultilabelAUROC)
    from utilities import load_pickle, read_json

    from data_collection import AudioSet

    sys.path.append("../src/lavis")
    from lavis.models.lam_models.swift import SWIFT

    def min_max_normalization(x, ori_min, ori_max, tgt_min, tgt_max):
        res = (x - ori_min) / (ori_max - ori_min)
        return (tgt_max - tgt_min) * res + tgt_min

    num_class = 527
    temp = torch.nn.Parameter(0.07 * torch.ones([]))  # 0.07
    metric = MetricCollection([
        MultilabelAccuracy(num_labels=num_class,
                           average="micro",
                           threshold=0.002),
        MultilabelF1Score(num_labels=num_class,
                          average="macro",
                          threshold=0.002),
        MultilabelAveragePrecision(num_labels=num_class, average="macro"),
        MultilabelAUROC(num_labels=num_class, average='macro')
    ])
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    swift = SWIFT()
    swift.init_swift(device=device)
    swift.load_checkpoint(
        # "/data/EECS-MachineListeningLab/jinhua/lam/check_point/Pretrain_stage2_audioset_aqa/without_boa.20230803203/checkpoint_150000.pth"
        # "/data/EECS-MachineListeningLab/jinhua/lam/check_point/Pretrain_stage1_audioset/20230728062/checkpoint_150000.pth"
        ckpt_path)

    # category_tokens = load_pickle("./categories.pickle")
    category_tokens = get_category_tokens(swift)

    # sim = swift.compute_similarity(category_tokens, category_tokens)
    # min, max = sim.min(), sim.max()

    data = read_json(
        "/data/home/eey340/WORKPLACE/LAM/engine/results/lam_on_audioset_val_new.json"
    )
    preds = {datum["file_name"]: datum["predicted_answer"] for datum in data}

    dataset = AudioSet.build_dataset({
        "dataset_name": "eval",
        "audio_processor_name": "null",
        "output_format": ['file_path', 'multihot']
    })

    for fpath, gt in dataset:
        text_emb = swift.get_text_embedding(
            preds[fpath],
            padding=False,
        )["text_embedding"].squeeze(0)

        sim, _ = swift.compute_similarity(text_emb, category_tokens).max(dim=0)

        # sim = min_max_normalization(
        #     sim,
        #     ori_min=min,
        #     ori_max=max,
        #     tgt_min=-1,
        #     tgt_max=1,
        # )
        # sim /= temp
        # sim = sim.sigmoid()
        metric.update(sim.unsqueeze(0), gt.unsqueeze(0).long())
        # break
    result = metric.compute()
    print(result)


if __name__ == "__main__":
    # generate_answer_on_audioset()
    ckpt_path = "/data/EECS-MachineListeningLab/jinhua/lam/check_point/Pretrain_stage2_audioset_aqa/20230814203/checkpoint_30000.pth"
    # get_category_tokens(ckpt_path)
    evaluate_on_audioset(ckpt_path)
