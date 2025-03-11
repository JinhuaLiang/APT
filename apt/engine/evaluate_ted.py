import os
import torch
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
from random import randint

from data_collection import AudioSet
from utilities import collate_func, set_logger, write_json, collate_func
from data_collection import AudioSet_SL
import sys

sys.path.append("../src/lavis")
from lavis.models import load_model_and_preprocess

log = set_logger(__name__)


def generate_answer_on_audioset(
    lam_ckpt_path="",
    results_json_path="/data/home/eey340/WORKPLACE/LAM/engine/results/lam_on_audioset_val_new2.json",
    mini_data=True,
):
    # Outline the given audio samples briefly. Summarize the audio with key words.
    prompt = "When the sound {} happens?"
    batch_size = 10

    dataset = AudioSet_SL.build_dataset(
        {
            "dataset_name": "eval",
            "start_time": 0,
            "end_time": 10,
            "output_format": [
                "file_path",
                "time_stamps",
                "fbank",
                "mids",
                "category",
                "ground_truth",
            ],
        },
    )

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    # loads lam pre-trained model
    model, _, _ = load_model_and_preprocess(
        name="lam_vicuna_instruct",
        # NOTE: change to other weights
        model_type="vicuna7b-ft",  # vicuna1.5_7b-ft
        is_eval=True,
        device=device,
    )

    model.load_from_pretrained(lam_ckpt_path)

    # load sample image
    # loader = DataLoader(
    #     dataset,
    #     batch_size=batch_size,
    #     collate_fn=collate_func,
    # )

    results = []
    with tqdm(total=10 if mini_data else len(dataset)) as pbar:
        for batch_idx, data in enumerate(dataset):
            mid2cat = dict(zip(data[3], data[4]))
            event_mid = data[3][randint(0, len(data[3]) - 1)]

            gt = []
            for stt, edt in data[5][event_mid]:
                gt.append(f"{stt}s-{edt}s")
            gt = "; ".join(gt) + "."

            output = model.generate(
                {
                    "audio": data[2].cuda().unsqueeze(0),
                    "prompt": prompt.format(mid2cat[event_mid]),
                },
                temperature=0.1,
            )

            res = {
                "file_name": data[0],
                "prediction": output,
                "ground_truth": gt,
            }
            log.info(res)
            results.append(res)

            # # output = torch.tensor_split(output, batch_size, dim=0)
            # for fname, predicted_answer, tgt_cat, tgt_multihot in zip(
            #     data[0], output, data[3], data[6]
            # ):
            #     res = {
            #         "file_name": fname,
            #         "predicted_answer": predicted_answer,
            #         "target_label": tgt_cat,
            #         "ground_truth": ",".join(
            #             [str(int(elem)) for elem in tgt_multihot.tolist()]
            #         ),
            #     }
            #     log.info(res)
            #     results.append(res)

            pbar.update(1)
            if mini_data and batch_idx == 10:
                break

    write_json(results, results_json_path)


if __name__ == "__main__":
    import argparse

    lam_ckpt_dir = (
        "/data/EECS-MachineListeningLab/jinhua/lam/check_point/Pretrain_stage2"
    )
    # "/data/EECS-MachineListeningLab/jinhua/lam/check_point/Pretrain_stage2_audioset_aqa/20230814203/checkpoint_30000.pth"
    results_json_dir = (
        "/data/home/eey340/WORKPLACE/LAM/src/lavis/run_scripts/lam/eval/results"
    )

    mini_data = False

    ckpt_foldername = "20230905031"  # args.ckpt_folder_name  # "20230829124"
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
        "checkpoint_2500000.pth",
        # "checkpoint_3750000.pth",
    ]:
        lam_ckpt_path = os.path.join(lam_ckpt_dir, ckpt_foldername, pth)
        results_json_path = os.path.join(
            results_json_dir,
            f"{ckpt_foldername}_{pth}_ted.json",
        )

        generate_answer_on_audioset(
            lam_ckpt_path=lam_ckpt_path,
            results_json_path=results_json_path,
            mini_data=mini_data,
        )
