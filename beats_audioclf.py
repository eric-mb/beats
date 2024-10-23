import sys
import torch
import pickle
import json
import logging
import argparse
import librosa
from pathlib import Path
from typing import Dict, Any, Tuple

sys.path.append("unilm/beats/")
from BEATs import BEATs, BEATsConfig

sys.path.append(".")
from project_utils import set_seeds, setup_logging


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
    return parser.parse_args()


def get_models(device: str, script_dir: Path) -> Tuple[BEATs, Dict[int, str]]:
    checkpoint = torch.load(
        script_dir / "pretrained_utils/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"
    )
    cfg = BEATsConfig(checkpoint["cfg"])
    BEATs_model = BEATs(cfg)
    BEATs_model.load_state_dict(checkpoint["model"])
    BEATs_model.to(device)
    BEATs_model.eval()

    with open(script_dir / "pretrained_utils/ontology.json", "r") as f:
        data = json.load(f)

    idx_to_code = {v: k for k, v in checkpoint["label_dict"].items()}

    label_map = {}
    for entry in data:
        if entry["id"] in idx_to_code:
            label_map[idx_to_code[entry["id"]]] = entry["name"]

    return BEATs_model, label_map


def process_video(
    audio_path: Path,
    beats_model: BEATs,
    label_map: Dict[int, str],
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

        audio_array, _ = chunk
        audio_array = torch.tensor(audio_array).unsqueeze(0).to(torch.float32)

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

        top3_label_probs = []
        for i, (top3_label_prob, top3_label_idx) in enumerate(zip(*probs.topk(k=3))):
            top3_label_probs.append(
                (
                    [label_map[label_idx.item()] for label_idx in top3_label_idx],
                    top3_label_prob.tolist(),
                )
            )

        predictions.append(
            {
                "start": start_time,
                "end": end_time,
                "top3_label": [l for l, _ in top3_label_probs],
                "top3_label_prob": [p for _, p in top3_label_probs],
            }
        )

    return predictions


def main():
    script_path = Path(__file__).resolve()
    log_file = setup_logging("audio_classification")

    args = parse_args()

    logging.info(f"Log file will be saved at: {log_file}")
    set_seeds(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    beats_model, label_map = get_models(device, script_path.parent)

    for video in args.videos:
        video_path = Path(video)
        audio_path = Path(args.pkl_dir) / "audio.wav"
        output_path = Path(args.pkl_dir) / video_path.stem / "audioClf.pkl"
        output_path.mkdir(parents=True, exist_ok=True)

        audio_cls = process_video(
            audio_path, video_path, beats_model, label_map, device
        )

        output_dict = {
            "github_repo": "https://github.com/microsoft/unilm",
            "commit_id": "13641268b59df5cf90d27b451d87ab58b6a07055",
            "parameters": "default",
            "video_file": str(video_path),
            "output_data": audio_cls,
        }

        with open(output_path, "wb") as f:
            pickle.dump(output_dict, f)

        logging.info(f"Saved shot audio classification to: {output_path}")


if __name__ == "__main__":
    main()
