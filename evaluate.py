import shutil
import sys
import glob
import yaml

import argparse
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from collections import defaultdict
from easydict import EasyDict as edict

from pytorch_msssim import ms_ssim
from models.envc import ENVCwAR
import models


def load_img_to_tensor(file_p):
    x = torch.from_numpy(np.array(Image.open(file_p)))
    x = x.permute(2, 0, 1).unsqueeze(0)
    return x


def save_tensor_to_img(x, file_p):
    assert x.shape[0] == 1
    x = x.squeeze(0)
    x = x.astype(np.uint8).transpose((1, 2, 0))
    Image.fromarray(x).save(file_p)


class Scalar:
    def __init__(self, name: str, value: float, precision: int):
        self.name = name
        self.value = round(value, precision)
        self.precision = precision

    def to_string(self):
        return f"{self.name} {self.value:.{self.precision}f}"


class Text:
    def __init__(self, name: str, value: str):
        self.name = name
        self.value = value

    def to_string(self):
        return f"{self.name} {self.value}"


class FrameLogger:
    def __init__(self, name, logs=None):
        self.name = name
        self.logs = [] if logs is None else logs

    def add_scalar(self, name, value, precision=6):
        self.logs.append(Scalar(name, value, precision))

    def add_text(self, name, value):
        self.logs.append(Text(name, value))

    def to_string(self):
        out = f"[{self.name}]" + " "
        for i, log in enumerate(self.logs):
            out += log.to_string()
            if i != len(self.logs) - 1:
                out += " "
        return out


class SequenceLogger(FrameLogger):
    def __init__(self, name, logs=None, loggers=None):
        super().__init__(name, logs)
        self.loggers = [] if loggers is None else loggers

    def average_scalar_over_loggers(self):
        scalar_accumulator = defaultdict(float)
        scalar_counter = defaultdict(int)
        scalar_precision = defaultdict(int)
        for logger in self.loggers:
            for log in logger.logs:
                if not isinstance(log, Scalar):
                    continue
                scalar_accumulator[log.name] += log.value
                scalar_counter[log.name] += 1
                scalar_precision[log.name] += log.precision

        for name, value in scalar_accumulator.items():
            prec = scalar_precision[name]
            count = scalar_counter[name]
            self.add_scalar(f"{name}", value / count , round(prec / count))


@torch.no_grad()
def evaluate(model, seq_ps, n_frames, intra_period, out_dir=None, log_level=2):
    model.eval()

    seq_out_dir = None
    seq_loggers = []
    for seq_p in seq_ps:
        seq_p = Path(seq_p)
        assert seq_p.is_dir()
        frame_ps = sorted(seq_p.iterdir())[:n_frames]

        if out_dir is not None:
            seq_out_dir = out_dir / seq_p.name

        seq_logger = evaluate_one_sequence(
            model, frame_ps, intra_period, seq_out_dir, log_level)
        seq_loggers.append(seq_logger)

    dataset_p = Path(seq_ps[0]).parent
    dataset_logger = SequenceLogger(
        name=f"Dataset '{dataset_p}'", loggers=seq_loggers)
    dataset_logger.average_scalar_over_loggers()
    print(dataset_logger.to_string())


@torch.no_grad()
def evaluate_one_sequence(
        model, frame_ps, intra_period, out_dir=None, log_level=2):
    if out_dir is not None:
        shutil.rmtree(out_dir, ignore_errors=True)

    loggers = []
    ref_buffer = {}
    frame_type_list = model.build_frame_type_list(len(frame_ps), intra_period)
    for idx in range(len(frame_ps)):
        frame_p = frame_ps[idx]
        frame_type = frame_type_list[idx]

        ori = load_img_to_tensor(frame_p).to(model.device)

        x = ori.to(torch.float32).div(255.)
        x = model.pre_padding(x)
        x_hat, bits_list, ref_buffer = model.forward_one_frame(
            x, ref_buffer, frame_type)

        hgt, wdt = ori.shape[2:4]
        rec = model.post_cropping(x_hat, wdt, hgt)
        rec = rec.clamp(0, 1).mul(255.).round().to(torch.uint8)

        # statistics
        ori = ori.float()
        rec = rec.float()
        num_pixels = hgt * wdt
        bpp = sum(bits_list).item() / num_pixels
        mse = ((ori - rec) ** 2).mean().item()
        psnr = 20 * np.log10(255.) - 10 * np.log10(mse)
        msssim = ms_ssim(ori, rec, data_range=255).item()

        seq_stem = frame_p.parent.stem
        logger = FrameLogger(name=f"Frame {seq_stem} {idx+1:0>6}")
        logger.add_text("type", frame_type)
        logger.add_scalar("bpp", bpp, precision=6)
        logger.add_scalar("psnr", psnr, precision=4)
        logger.add_scalar("msssim", msssim, precision=6)
        loggers.append(logger)
        if log_level >= 2:
            print(logger.to_string())

        # save reconstruction
        if out_dir is not None:
            rec = rec.cpu().numpy()
            suffix = ".png"
            frame_stem = [
                frame_p.stem,
                f"bpp{bpp:.4f}",
                f"psnr{psnr:.2f}",
                f"msssim{msssim:.4f}"
            ]
            frame_stem = "_".join(frame_stem)
            rec_name = frame_stem + suffix
            rec_p = out_dir / rec_name
            rec_p.parent.mkdir(parents=True, exist_ok=True)
            save_tensor_to_img(rec, rec_p)

    seq_p = Path(frame_ps[0]).parent
    seq_logger = SequenceLogger(name=f"Sequence '{seq_p}'", loggers=loggers)
    seq_logger.average_scalar_over_loggers()
    if log_level >= 1:
        print(seq_logger.to_string())
    return seq_logger


def setup_args(parser):
    # required args
    parser.add_argument(
         "-c", "--config_path", type=Path, required=True,
        help="configuration file."
    )
    parser.add_argument(
        "-i", "--seq_glob", type=str, required=True,
        help="sequence directories that contain RGB frames in glob pattern"
    )
    parser.add_argument(
        "--ckpt_path", type=str, required=True,
        help="checkpoint path"
    )

    # optional args
    parser.add_argument(
        "-o", "--output_dir", type=Path,
        help="output directory for reconstruction [optional]"
    )
    parser.add_argument(
        "-f", "--n_frames", type=int, default=100,
        help="specifies the number of frames to be encoded [optional]"
    )
    parser.add_argument(
        "-ip", "--intra_period", type=int, default=12,
        help="the intra frame period [optional]"
    )
    parser.add_argument(
        "--log_level", type=int, choices=[0, 1, 2], default=2,
        help="the verbosity of the log [optional]"
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    setup_args(parser)
    args = parser.parse_args(sys.argv[1:])
    cfg = yaml.load(open(args.config_path, 'r'), Loader=yaml.FullLoader)
    cfg = edict(cfg)

    model = getattr(models, cfg.model.name)(**cfg.model.kwargs).cuda().eval()
    model.load_state_dict(torch.load(args.ckpt_path), strict=False)

    seq_ps = sorted(glob.glob(args.seq_glob))
    evaluate(model, seq_ps, args.n_frames, args.intra_period, args.output_dir,
             args.log_level)
