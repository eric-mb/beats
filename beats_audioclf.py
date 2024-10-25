import sys
import torch
import pickle
import json
import logging
import argparse
import librosa
from pathlib import Path
from typing import Dict, Any, Tuple
import os
import subprocess

from beats.BEATs import BEATs, BEATsConfig

sys.path.append(".")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Runs audio classification on shot segments"
    )
    parser.add_argument(
        "-v",
        "--videos",
        nargs="+",
        type=str,
        required=True,
        help="Path to input videos",
    )
    parser.add_argument(
        "-o", "--pkl_dir", type=str, required=True, help="Path to pkl directory"
    )
    parser.add_argument("--workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--rewrite", action="store_true", help="Force rewrite")
    parser.add_argument("--debug", action="store_true", help="Debugging outputs")
    return parser.parse_args()


def get_models(device: str) -> Tuple[BEATs, Dict[int, str]]:
    checkpoint = torch.load(
        Path(
            os.path.join(
                "beats",
                "checkpoints",
                "BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt",
            )
        )
    )
    cfg = BEATsConfig(checkpoint["cfg"])
    BEATs_model = BEATs(cfg)
    BEATs_model.load_state_dict(checkpoint["model"])
    BEATs_model.to(device)
    BEATs_model.eval()

    with open(os.path.join("beats", "checkpoints", "ontology.json"), "r") as f:
        data = json.load(f)

    idx_to_code = {v: k for k, v in checkpoint["label_dict"].items()}

    label_map = {}
    for entry in data:
        if entry["id"] in idx_to_code:
            label_map[idx_to_code[entry["id"]]] = entry["name"]
            if entry["name"] == "Music":
                music_id = idx_to_code[entry["id"]]

    return BEATs_model, label_map, music_id


def extract_audio(video_path: Path, output_path: Path, num_workers: int) -> bool:

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-qscale:a",
        "0",
        "-ac",
        "1",
        "-vn",
        "-threads",
        str(num_workers),
        "-ar",
        "16000",
        str(output_path),
        "-loglevel",
        "panic",
    ]

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to extract audio from {video_path}: {e}")
        return False


def process_video(
    audio_path: Path,
    beats_model: BEATs,
    label_map: Dict[int, str],
    music_id: int,
    device: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Takes 10sec audio chunks and classifies them into audio categories

    Args:
        audio_path (Path): Path to audio file
        beats_model (BEATs): BEATs model
        label_map (Dict[int, str]): Label map for BEATs model
        device (str): Device to run the model on

    Returns:
        List[Dict[str, Any]]: List of segment predictions
    """

    required_sr = 16000
    y, sr = librosa.load(audio_path, sr=None)

    # chunk duration 10 seconds
    chunk_duration = 10
    chunk_samples = int(chunk_duration * sr)
    chunks = [y[i : i + chunk_samples] for i in range(0, len(y), chunk_samples)]

    predictions = []
    for i, chunk in enumerate(chunks):
        start_time, end_time = i * 10, (i + 1) * 10

        audio_array = torch.tensor(chunk).unsqueeze(0).to(torch.float32)

        if audio_array.shape[1] < required_sr:
            audio_array = torch.nn.functional.pad(
                audio_array, (0, required_sr - audio_array.shape[1])
            )

        audio_array = audio_array.to(device)

        padding_mask = (
            torch.zeros(audio_array.shape[0], audio_array.shape[1]).bool().to(device)
        )

        with torch.no_grad():
            probs = beats_model.extract_features(
                audio_array, padding_mask=padding_mask
            )[0]

        predicted_labels = []
        for idx, entry in enumerate(probs.tolist()[0]):
            if entry > 0.3:
                predicted_labels.append(label_map[idx])

        top3_prob, top3_idx = probs.topk(k=3)
        top3_labels = [label_map[label_idx] for label_idx in top3_idx.tolist()[0]]
        predictions.append(
            {
                "start": start_time,
                "end": end_time,
                "top3_label": top3_labels,
                "top3_label_prob": top3_prob.tolist()[0],
                "predicted_labels": predicted_labels,
                "music_prob": probs.tolist()[0][music_id],
            }
        )

    return predictions


def main():
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    beats_model, label_map, music_id = get_models(device)

    # loop trough input videos
    for video in args.videos:
        # setup paths
        video_path = Path(video)
        audio_path = Path(args.pkl_dir) / video_path.stem / "audio.wav"
        output_path = Path(args.pkl_dir) / video_path.stem / "audioClf.pkl"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # check if audio file exists
        audio = True
        if not os.path.isfile(audio_path) or args.rewrite:
            # if not, create audio file
            logging.info(f"Extract audio for {video}")
            audio = extract_audio(
                video_path=video_path,
                output_path=audio_path,
                num_workers=args.workers,
            )

        # audio could not be loaded or extracted
        if not audio:
            logging.error(f"{audio_path} does not exist")
            continue

        # perform audio classification
        logging.info(f"Perform audio classification for {video}")
        audio_cls = process_video(audio_path, beats_model, label_map, music_id, device)

        # write output dict to pkl
        output_dict = {
            "github_repo": "https://github.com/microsoft/unilm",
            "commit_id": "13641268b59df5cf90d27b451d87ab58b6a07055",
            "parameters": "default",
            "video_file": str(video_path),
            "output_data": audio_cls,
        }

        with open(str(output_path), "wb") as f:
            pickle.dump(output_dict, f)

        logging.info(f"Saved audio classification to: {str(output_path)}")


if __name__ == "__main__":
    main()
