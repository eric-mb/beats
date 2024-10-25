"""Script Description"""

import argparse
import logging
import os
import pickle
from pympi.Elan import Eaf, to_eaf
import sys


def parse_args() -> dict:
    """Function to parse script arguments"""
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-v",
        "--videos",
        nargs="+",
        type=str,
        required=True,
        help="Path to input videos",
    )
    parser.add_argument(
        "-p", "--pkl_dir", type=str, required=True, help="Path to pkl directory"
    )
    parser.add_argument(
        "-e", "--eaf_dir", type=str, required=True, help="Path to eaf directory"
    )

    parser.add_argument("-vv", "--debug", action="store_true", help="debug output")
    args = parser.parse_args()
    return args


def create_shotdetection_tier(eaf, data, tiername):
    eaf.add_tier(tier_id=tiername)
    for i, shot in enumerate(data["output_data"]["shots"]):
        eaf.add_annotation(
            id_tier=tiername,
            start=int(1000 * shot["start"]),
            end=int(1000 * shot["end"]),
            value=f"Shot {str(i)}",
        )

    return eaf


def create_whisperx_tier(eaf, data, tiername):
    eaf.add_tier(tier_id=tiername)
    for st in data["output_data"]["speaker_turns"]:
        if st["start"] > st["end"]:
            continue

        eaf.add_annotation(
            id_tier=tiername,
            start=int(1000 * st["start"]),
            end=int(1000 * st["end"]),
            value=st["text"],
        )

    return eaf


def create_audioClf_tier(eaf, data, tiername):
    if "top3_label" in data["output_data"][0]:
        eaf.add_tier(tier_id=tiername + "_top3")
    if "predicted_labels" in data["output_data"][0]:
        eaf.add_tier(tier_id=tiername + "_predicted")
    if "music_prob" in data["output_data"][0]:
        eaf.add_tier(tier_id=tiername + "_music")

    for result in data["output_data"]:

        if "top3_label" in result and result["top3_label"]:
            eaf.add_annotation(
                id_tier=tiername + "_top3",
                start=int(1000 * result["start"]),
                end=int(1000 * result["end"]),
                value=", ".join(result["top3_label"]),
            )
        if "predicted_labels" in result and result["predicted_labels"]:
            eaf.add_annotation(
                id_tier=tiername + "_predicted",
                start=int(1000 * result["start"]),
                end=int(1000 * result["end"]),
                value=", ".join(result["predicted_labels"]),
            )
        if "music_prob" in result and result["music_prob"]:
            eaf.add_annotation(
                id_tier=tiername + "_music",
                start=int(1000 * result["start"]),
                end=int(1000 * result["end"]),
                value=str(result["music_prob"]),
            )

    return eaf


def main() -> int:
    """main function"""
    # load arguments
    args = parse_args()

    # define logging level and format
    level = logging.INFO
    if args.debug:
        level = logging.DEBUG

    logging.basicConfig(
        format="%(asctime)s %(levelname)s:%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=level,
    )

    for video in args.videos:
        # setup paths
        vidname, ext = os.path.splitext(os.path.basename(video))
        pkl_path = os.path.join(args.pkl_dir, vidname)

        # setup eaf file
        eaf = Eaf(author="TIB")
        eaf.remove_tier("default")
        eaf.add_linked_file(file_path=f"{vidname}{ext}", mimetype="video/mp4")

        # convert pkl to eaf
        for f in os.listdir(pkl_path):
            if os.path.splitext(f)[-1] != ".pkl":
                continue

            logging.info(f"Loading {os.path.join(pkl_path, f)}")

            with open(os.path.join(pkl_path, f), "rb") as pkl:
                data = pickle.load(pkl)

            if f == "transnet_shotdetection.pkl":
                eaf = create_shotdetection_tier(eaf, data, tiername="Shots")

            if f == "asr_whisperx.pkl":
                eaf = create_whisperx_tier(eaf, data, tiername="Speech")

            if f == "shot_audioClf.pkl":
                eaf = create_audioClf_tier(
                    eaf, data, tiername="Shot - AudioClassification"
                )

            if f == "whisperxspeaker_audioClf.pkl":
                eaf = create_audioClf_tier(
                    eaf, data, tiername="Speaker - AudioClassification"
                )

            if f == "audioClf.pkl":
                eaf = create_audioClf_tier(
                    eaf, data, tiername="10sec - AudioClassification"
                )

        # write eaf
        out_file = os.path.join(args.eaf_dir, f"{vidname}.eaf")
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        logging.info(f"Writing {out_file}")
        to_eaf(file_path=out_file, eaf_obj=eaf)

    return 0


if __name__ == "__main__":
    sys.exit(main())
