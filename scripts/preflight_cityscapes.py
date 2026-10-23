"""Run before paying for a full Cityscapes training job; no full training is started."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image
import torch
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--split', default='1_16')
    parser.add_argument('--batch-size', default=1, type=int)
    parser.add_argument('--steps', default=40, type=int)
    parser.add_argument('--memory-fraction', default=.85, type=float)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    os.chdir(repo)
    out = Path(args.output).resolve()
    if out.exists():
        raise ValueError('Use a new output directory; previous diagnostic runs are preserved')
    out.mkdir(parents=True)
    result = dict(status='running', started=time.strftime('%Y-%m-%d %H:%M:%S'), arguments=vars(args))
    try:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable: do not start formal training')
        if args.steps < 20 or args.batch_size < 1 or not 0 < args.memory_fraction <= 1:
            raise ValueError('Need >=20 steps, positive batch size, memory fraction in (0,1]')
        torch.cuda.init()
        result.update(gpu=torch.cuda.get_device_name(), torch_version=torch.__version__, cuda_version=torch.version.cuda,
                      bf16=torch.cuda.is_bf16_supported(), gpu_total_GiB=torch.cuda.get_device_properties(0).total_memory/2**30)
        root = Path(args.data_root).resolve()
        files = [repo/f'splits/cityscapes/{args.split}/labeled.txt', repo/f'splits/cityscapes/{args.split}/unlabeled.txt',
                 repo/'splits/cityscapes/val.txt']
        image_sets = []
        for file in files:
            lines = [line.split() for line in file.read_text().splitlines() if line.strip()]
            if not lines or any(len(row)!=2 for row in lines):
                raise ValueError(f'Invalid split: {file}')
            image_sets.append({row[0] for row in lines})
            for row in lines:
                for relative in row:
                    path=root/relative
                    if not path.is_file() or not path.stat().st_size:
                        raise FileNotFoundError(path)
            result[file.as_posix()] = len(lines)
        if image_sets[0]&image_sets[1] or (image_sets[0]|image_sets[1])&image_sets[2]:
            raise ValueError('Labeled/unlabeled/validation image overlap')
        first = files[0].read_text().splitlines()[0].split()
        with Image.open(root/first[1]) as im:
            ids = np.unique(np.asarray(im))
        if not np.isin(ids,list(range(19))+[255]).all():
            raise ValueError('Expected Cityscapes labelTrainIds, not raw labelIds')
        if not (repo/'pretrained/dinov2_base.pth').is_file():
            raise FileNotFoundError('pretrained/dinov2_base.pth')
        cfg = yaml.safe_load((repo/'configs/cityscapes_du_match.yaml').read_text())
        cfg['data_root']=str(root)
        cfg['batch_size']=args.batch_size
        cfg['du_match']['warmup_iters']=10  # Timing only, never the formal warm-up schedule.
        config=out/'timing_config.yaml'
        config.write_text(yaml.safe_dump(cfg,sort_keys=False))
        run=out/'training'
        command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=1','du_match.py',
            '--config',str(config),'--labeled-id-path',str(files[0]),'--unlabeled-id-path',str(files[1]),
            '--save-path',str(run),'--benchmark-steps',str(args.steps),'--benchmark-discard','5',
            '--torch-sdpa','--benchmark-memory-fraction',str(args.memory_fraction)]
        if result['bf16']:
            command.append('--amp-bf16')
        result['command']=command
        print(json.dumps(result,indent=2),flush=True)
        with (out/'training.log').open('w',buffering=1) as log:
            proc=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            for line in proc.stdout:
                print(line,end='',flush=True)
                log.write(line)
            result['training_exit_code']=proc.wait()
        if result['training_exit_code']:
            raise RuntimeError('Training preflight failed; inspect training.log before spending on a full run')
        benchmark=json.loads((run/'benchmark.json').read_text())
        if 'du_match_active' not in benchmark['phases'] or not all(math.isfinite(r['loss']) for r in benchmark['records']):
            raise RuntimeError('No successful active phase or nonfinite losses')
        result.update(status='passed_short_training_only',benchmark=benchmark,
            next_checks='Before formal results: verify checkpoint save/resume and official sliding-window validation; use identical precision/attention settings for baseline and DU-Match.')
    except Exception as error:
        result.update(status='failed',error=repr(error))
        raise
    finally:
        (out/'preflight.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
