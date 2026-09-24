"""WSL-side broker for the fine-tuned 24B option brain (RTX 5090 only): one forward pass, argmax over legal letters."""
import argparse, json, os, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import option_sft as S  # sets CUDA_VISIBLE_DEVICES to the 5090 before torch import
import torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--adapter', default=None)
    ap.add_argument('--bus', required=True)
    a = ap.parse_args()
    bus = Path(a.bus); (bus/'requests').mkdir(parents=True, exist_ok=True); (bus/'responses').mkdir(parents=True, exist_ok=True)
    codec = S.Codec(); letter = codec.letter_ids()
    model = S.load_model(a.adapter)
    (bus/'broker-ready.json').write_text(json.dumps(dict(model='mistral24b-nf4', adapter=a.adapter, gpu=torch.cuda.get_device_name(0), pid=os.getpid(), time=time.time())))
    served = 0
    while not (bus/'STOP').exists():
        reqs = sorted((bus/'requests').glob('*.json'))
        if not reqs:
            time.sleep(0.02); continue
        for p in reqs:
            try: req = json.loads(p.read_text())
            except Exception: continue
            t0 = time.time()
            ids = codec.encode([dict(role='system', content=req['system']), dict(role='user', content=req['user'])])
            x = torch.tensor([ids], device='cuda')
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False).logits[0, -1].float()
            labels = req['labels']
            pr = torch.softmax(torch.stack([logits[letter[L]] for L in labels]), 0).tolist()
            k = max(range(len(labels)), key=lambda i: pr[i])
            res = dict(id=req['id'], choice=labels[k], confidence=pr[k], probs={labels[i]: pr[i] for i in sorted(range(len(labels)), key=lambda i: -pr[i])[:8]},
                       prompt_n=len(ids), seconds=time.time() - t0, model='mistral24b-nf4' + ('+lora' if a.adapter else ''))
            tmp = bus/'responses'/(p.stem+'.tmp'); tmp.write_text(json.dumps(res)); os.replace(tmp, bus/'responses'/(p.stem+'.json')); p.unlink(); served += 1
    (bus/'broker-stopped.json').write_text(json.dumps(dict(served=served, time=time.time())))

if __name__ == '__main__':
    main()
