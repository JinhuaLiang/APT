import torch
import argparse
from PIL import Image
import sys

sys.path.append("../src/lavis")
from lavis.models import load_model_and_preprocess


def parse_args():
    parser = argparse.ArgumentParser(
        description='Easy audio feature extractor')

    parser.add_argument(
        "--csv_dir",
        type=str,
        help=
        "Should be a csv file of a single columns, each row is the input video path."
    )
    parser.add_argument("--audio_output_dir",
                        type=str,
                        help="The place to store the video frames.")
    parser.add_argument("--mini_data",
                        action='store_true',
                        default=False,
                        help="Test on mini_batch.")
    args = parser.parse_args()

    return args


def main():
    # setup device to use
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    # load sample image
    raw_images = [
        Image.open(
            f"/data/EECS-MachineListeningLab/datasets/ACAV/acav200k/frames/custom_ncentroids-500-subset_size-200K_part00000/frame_{i}/YTk8amj7JnYyE.jpg"
        ).convert("RGB") for i in range(10)
    ]
    # print(raw_image.resize((596, 437)))

    # loads BLIP-2 pre-trained model
    model, vis_processors, _ = load_model_and_preprocess(
        name="blip2_t5",
        model_type="pretrain_flant5xl",
        is_eval=True,
        device=device)
    # prepare the image
    # image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    image = [
        vis_processors["eval"](img).unsqueeze(0).to(device)
        for img in raw_images
    ]

    # answer1 = model.generate({
    #     "image":
    #     image,
    #     "prompt":
    #     "Question: what is in the image? Answer:"  # which city is this?
    # })
    answer1 = model.video_as_imput({
        "image":
        image,
        "prompt":
        "Question: what is in the image? Answer:"  # which city is this?
    })
    print(answer1)


if __name__ == "__main__":
    # args = parse_args()
    # main(args)
    main()
