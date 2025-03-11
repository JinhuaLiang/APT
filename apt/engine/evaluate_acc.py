import os
import torch
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_collection import AudioCaps, Clotho
from utilities import collate_func, set_logger, write_json, collate_func
import sys

sys.path.append("../src/lavis")
from lavis.models import load_model_and_preprocess

log = set_logger(__name__, level="debug")


def generate_answer_on_audio_captioning_dataset(
    dataset,
    lam_ckpt_path="",
    results_json_path="./test.json",
    mini_data=True,
):
    prompt = "Describe the audio clip concisely."
    batch_size = 10
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    # loads lam pre-trained model
    model, _, _ = load_model_and_preprocess(
        name="lam_vicuna_instruct",  # lam_vicuna_instruct_linear_beats | lam_vicuna_instruct_linear_clap | lam_vicuna_instruct_linear | lam_vicuna_instruct
        # NOTE: change to other weights
        model_type="vicuna7b-ft",  # vicuna7b | vicuna7b-ft | vicuna1.5_7b | vicuna1.5_7b-ft
        is_eval=True,
        device=device,
    )

    model.load_from_pretrained(lam_ckpt_path)

    # load sample image
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_func,
    )

    results = []
    with tqdm(total=10 if mini_data else len(loader)) as pbar:
        for batch_idx, data in enumerate(loader):
            # NOTE: change to gererate_new if bos is in the first position.
            output = model.generate(
                {
                    "audio": data[1].cuda(),
                    "prompt": prompt,
                },
                num_captions=1,
                max_length=256,
                min_length=1,
                repetition_penalty=1.5,
                length_penalty=1,
                # beam searching
                num_beams=5,  # 1 | 5
                # top-p sampling
                use_nucleus_sampling=False,  # True | False
                temperature=None,  # 0.1 | None
                top_p=None,  # 0.9 | None
            )
            print(output)
            # output = torch.tensor_split(output, batch_size, dim=0)
            for fname, predicted_answer, ground_truth in zip(data[0], output, data[2]):
                res = {
                    "file_name": fname,
                    "predicted_answer": predicted_answer,
                    "ground_truth": ground_truth,
                }
                # log.info(res)
                results.append(res)

            pbar.update(1)
            if mini_data and batch_idx == 10:
                break

    write_json(results, results_json_path)


def evaluate_on_captioning_dataset(results_json_path):
    from aac_metrics import evaluate
    from utilities import read_json

    data = read_json(results_json_path)

    preds, tgts = [], []
    for _, datum in enumerate(data):
        preds.append(datum["predicted_answer"].replace("\n", ". "))
        tgts.append(datum["ground_truth"])

    results, _ = evaluate(preds, tgts)

    log.info(f"Finish evaluate {results_json_path}")
    log.info(results)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-ckpt", "--checkpoint-path", type=str)
    parser.add_argument("-o", "--output-dir", type=str)
    parser.add_argument("--mini-data", default=False, action="store_true")
    args = parser.parse_args()

    results_json_dir = args.output_dir
    os.makedirs(results_json_dir, exist_ok=True)
    mini_data = args.mini_data
    lam_ckpt_path = args.checkpoint_path

    for dataset_cls in (
        AudioCaps,
        Clotho,
    ):
        dataset = dataset_cls.build_dataset(
            {
                "dataset_name": "eval",  # eval
                "output_format": [
                    "file_path",
                    "fbank",
                    "caption",
                ],
            }
        )

        pth = lam_ckpt_path.split("/")[-1].split(".")[0]
        results_json_path = os.path.join(
            results_json_dir, f"{dataset.__class__.__name__}_{pth}.json"
        )

        if not os.path.exists(results_json_path):
            generate_answer_on_audio_captioning_dataset(
                dataset,
                lam_ckpt_path=lam_ckpt_path,
                results_json_path=results_json_path,
                mini_data=mini_data,
            )

        evaluate_on_captioning_dataset(results_json_path=results_json_path)
        # try:
        #     evaluate_on_captioning_dataset(results_json_path=results_json_path)
        # except:
        #     log.warning("Cannot calculate aac metric in this script.")
