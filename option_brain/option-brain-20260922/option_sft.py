"""Option-label QLoRA for Mistral-Small-3.2-24B (local BF16 base, NF4) + single-pass letter evaluation.

Modes:
  eval  --rows ... [--adapter DIR]            score rows: argmax over legal option letters at the first assistant token
  train --train-source king|opening|all       LoRA on option letters (assistant-token loss only), then eval on the other world
Runs in the WSL venv on the RTX 5090 only. Base weights are local; nothing is downloaded.
"""
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

GPU5090 = '$AD_GPU_UUID'
os.environ['CUDA_VISIBLE_DEVICES'] = GPU5090
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from render import SYSTEM_V2 as SYSTEM  # noqa: E402

BASE = Path('$AD_HOME/strategy_brain_sft_20260922/base-mistral24/weights')
REL = Path('$AD_ROOT/relabel')
OUT = Path('$AD_ROOT/sft')

import torch  # noqa: E402
from mistral_common.protocol.instruct.messages import SystemMessage, UserMessage, AssistantMessage  # noqa: E402
from mistral_common.protocol.instruct.request import ChatCompletionRequest  # noqa: E402
from mistral_common.protocol.instruct.validator import ValidationMode  # noqa: E402
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer  # noqa: E402


class Codec:
    def __init__(self):
        p = BASE / 'tekken.json'
        self.train = MistralTokenizer.from_file(p, mode=ValidationMode.finetuning)
        self.serve = MistralTokenizer.from_file(p, mode=ValidationMode.test)

    def encode(self, messages, training=False):
        cls = {'system': SystemMessage, 'user': UserMessage, 'assistant': AssistantMessage}
        req = ChatCompletionRequest(messages=[cls[m['role']](content=m['content']) for m in messages])
        return (self.train if training else self.serve).encode_chat_completion(req).tokens

    def example(self, user, letter):
        msgs = [dict(role='system', content=SYSTEM), dict(role='user', content=user), dict(role='assistant', content=letter)]
        prompt = self.encode(msgs[:-1])
        whole = self.encode(msgs, True)
        if whole[:len(prompt)] != prompt:
            raise ValueError('prompt prefix mismatch')
        return dict(input_ids=whole, labels=[-100] * len(prompt) + whole[len(prompt):], prompt=prompt)

    def letter_ids(self):
        ids = {}
        from options import LABELS
        for L in LABELS:
            ex = self.example('x', L)
            tail = ex['input_ids'][len(ex['prompt']):]
            ids[L] = tail[0]
        if len(set(ids.values())) != len(ids):
            raise RuntimeError('letters do not map to distinct first tokens')
        return ids


def load_model(adapter=None, train=False):
    from transformers import AutoModelForImageTextToText, BitsAndBytesConfig
    torch.set_num_threads(4)
    base = AutoModelForImageTextToText.from_pretrained(str(BASE), local_files_only=True, dtype=torch.bfloat16, device_map={'': 0},
                                                       attn_implementation='sdpa',
                                                       quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                                                                              bnb_4bit_compute_dtype=torch.bfloat16,
                                                                                              bnb_4bit_use_double_quant=True))
    if '5090' not in torch.cuda.get_device_name(0):
        raise RuntimeError('must run on the RTX 5090')
    from peft import prepare_model_for_kbit_training, get_peft_model, LoraConfig, PeftModel
    if adapter:
        model = PeftModel.from_pretrained(base, adapter, is_trainable=False)
        model.eval()
        return model
    if not train:
        base.eval()
        return base
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False})
    targets = [n for n, m in base.named_modules() if 'language_model' in n and n.endswith(('.q_proj', '.k_proj', '.v_proj', '.o_proj'))]
    model = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias='none', target_modules=targets, task_type='CAUSAL_LM'))
    model.config.use_cache = False
    return model


REL_DIR = [REL]


def rel_facts_version(rel_dir):
    """Facts version of a relabel directory's prompts: 'facts_version' of its facts-report.json (rerender.py records it
    since round 9), else None (no report: plain prompts or an older layout; or a report from before round 9)."""
    p = Path(rel_dir) / 'facts-report.json'
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding='utf-8')).get('facts_version')


def rows_for(source):
    # 'all' = both Astra worlds; 'a+b+c' = several row files in REL_DIR (e.g. all+dagger1).
    srcs = []
    for part in source.split('+'):
        srcs += ['opening', 'king'] if part == 'all' else [part]
    return [json.loads(l) for s in srcs for l in open(REL_DIR[0] / f'{s}.jsonl') if json.loads(l)['label']]


@torch.no_grad()
def evaluate(model, codec, rows, out_path):
    letter = codec.letter_ids()
    model.eval()
    preds = []
    t0 = time.time()
    with open(out_path, 'w') as fh:
        for r in rows:
            prompt = codec.encode([dict(role='system', content=SYSTEM), dict(role='user', content=r['prompt'])])
            x = torch.tensor([prompt], device='cuda')
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False).logits[0, -1].float()
            labels = [o['label'] for o in r['options']]
            z = torch.stack([logits[letter[L]] for L in labels])
            p = torch.softmax(z, 0).tolist()
            k = max(range(len(labels)), key=lambda i: p[i])
            pred = dict(source=r['source'], number=r['number'], label=r['label'], acceptable=r['acceptable'], label_kind=r['label_kind'],
                        choice=labels[k], choice_kind=r['options'][k]['kind'], confidence=p[k],
                        probs={labels[i]: p[i] for i in sorted(range(len(labels)), key=lambda i: -p[i])[:6]})
            preds.append(pred)
            fh.write(json.dumps(pred) + '\n')
    ok = sum(p['choice'] in p['acceptable'] for p in preds)
    kinds = {}
    for p in preds:
        kinds.setdefault(p['label_kind'], [0, 0])
        kinds[p['label_kind']][1] += 1
        kinds[p['label_kind']][0] += p['choice'] in p['acceptable']
    return dict(n=len(preds), acc=round(ok / len(preds), 4), seconds=round(time.time() - t0, 1),
                mean_conf=round(sum(p['confidence'] for p in preds) / len(preds), 4),
                by_kind={k: f'{a}/{b}' for k, (a, b) in sorted(kinds.items(), key=lambda kv: -kv[1][1])})


def check_option_block(row):
    """PLAN-annotations-v4.md section 5 (option_sft.py): permuted() keeps the prompt up to 'Options:' (the fact lines
    included) and rebuilds everything after it from row['options'] as '{label}. {text}' lines plus 'Answer with one
    letter.'. So that the shuffled copy loses nothing but the order, the prompt's own tail must be exactly that: a
    suffix added to an option line in the prompt only, or any line after the options (a fact line put there), would
    silently vanish from the shuffled copy. Raises AssertionError (also under python -O) when it is not."""
    nl = chr(10)
    head, mark, tail = row['prompt'].partition(nl + 'Options:' + nl)
    want = nl.join(f"{o['label']}. {o['text']}" for o in row['options']) + nl + 'Answer with one letter.'
    if not mark or tail != want:
        rid = row.get('id') or (row.get('source'), row.get('number'))
        raise AssertionError(f"option_sft.permuted: the option block of row {rid} is not the one rebuilt from row['options'] "
                             f"(the shuffled copy would drop or change text); prompt tail {tail[-300:]!r}")
    return head


def permuted(row, rng):
    """Same decision with the option list shuffled and re-lettered (position-bias augmentation)."""
    from options import LABELS
    nl = chr(10)
    head = check_option_block(row)
    opts = list(row['options'])
    rng.shuffle(opts)
    remap = {o['label']: LABELS[i] for i, o in enumerate(opts)}
    lines = [f"{LABELS[i]}. {o['text']}" for i, o in enumerate(opts)]
    prompt = head + nl + 'Options:' + nl + nl.join(lines) + nl + 'Answer with one letter.'
    return prompt, remap[row['label']]


def train(model, codec, rows, out, epochs=1, lr=1e-4, accum=8, seed=20260922, copies=2):
    random.seed(seed)
    torch.manual_seed(seed)
    rng = random.Random(seed + 7)
    data = []
    for r in rows:
        data.append(codec.example(r['prompt'], r['label']))
        for _ in range(copies - 1):
            prompt, label = permuted(r, rng)
            data.append(codec.example(prompt, label))
    trainables = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainables, lr=lr)
    steps = 0
    journal = open(out / 'updates.jsonl', 'w')
    model.train()
    for ep in range(epochs):
        order = list(range(len(data)))
        random.Random(seed + ep).shuffle(order)
        for off in range(0, len(order), accum):
            batch = order[off:off + accum]
            opt.zero_grad(set_to_none=True)
            losses = []
            for i in batch:
                x = torch.tensor([data[i]['input_ids']], device='cuda')
                y = torch.tensor([data[i]['labels']], device='cuda')
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss = model(input_ids=x, attention_mask=torch.ones_like(x), labels=y, use_cache=False).loss
                if not torch.isfinite(loss):
                    raise FloatingPointError('nonfinite loss')
                (loss / len(batch)).backward()
                losses.append(float(loss))
            norm = float(torch.nn.utils.clip_grad_norm_(trainables, 1.0))
            opt.step()
            steps += 1
            item = dict(step=steps, epoch=ep + 1, loss=sum(losses) / len(losses), grad_norm=norm, time=time.time())
            journal.write(json.dumps(item) + '\n')
            journal.flush()
            print(json.dumps(item), flush=True)
    journal.close()
    model.save_pretrained(out / 'adapter', safe_serialization=True)
    return steps


@torch.no_grad()
def val_score(model, codec, rows):
    """Mean letter NLL and accuracy over legal letters (eval mode); used for checkpoint selection."""
    letter = codec.letter_ids()
    was_training = model.training
    model.eval()
    nll, hits = 0.0, 0
    for r in rows:
        ids = codec.encode([dict(role='system', content=SYSTEM), dict(role='user', content=r['prompt'])])
        x = torch.tensor([ids], device='cuda')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            lg = model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False).logits[0, -1].float()
        labels = [o['label'] for o in r['options']]
        p = torch.softmax(torch.stack([lg[letter[L]] for L in labels]), 0)
        nll += float(-torch.log(p[labels.index(r['label'])] + 1e-9))
        hits += labels[int(p.argmax())] in r['acceptable']
    if was_training:
        model.train()
    return nll / len(rows), hits / len(rows)


def train_stable(model, codec, rows, out, epochs=1, lr=1e-4, accum=8, seed=20260922, copies=2, val_frac=0.04, val_every=60,
                 warmup=20, floor=0.1):
    """Same examples and batching as train(), plus: a seeded validation slice of rows held out from training,
    warmup + cosine learning-rate decay to floor*lr, validation every val_every updates, and the best
    checkpoint (lowest validation NLL) restored before saving. Added 2026-09-22 after the d6plain run ended
    with a degraded model despite normal training losses."""
    import math
    random.seed(seed)
    torch.manual_seed(seed)
    idx = list(range(len(rows)))
    random.Random(seed + 99).shuffle(idx)
    n_val = max(24, int(len(rows) * val_frac))
    val_rows = [rows[i] for i in idx[:n_val]]
    train_rows = [rows[i] for i in idx[n_val:]]
    rng = random.Random(seed + 7)
    data = []
    for r in train_rows:
        data.append(codec.example(r['prompt'], r['label']))
        for _ in range(copies - 1):
            prompt, label = permuted(r, rng)
            data.append(codec.example(prompt, label))
    trainables = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainables, lr=lr)
    total = epochs * math.ceil(len(data) / accum)
    steps = 0
    journal = open(out / 'updates.jsonl', 'w')
    vlog = open(out / 'val.jsonl', 'w')
    best = None
    model.train()

    def checkpoint(step):
        nonlocal best
        v_nll, v_acc = val_score(model, codec, val_rows)
        vlog.write(json.dumps(dict(step=step, val_nll=round(v_nll, 4), val_acc=round(v_acc, 4), time=time.time())) + '\n')
        vlog.flush()
        print(json.dumps(dict(step=step, val_nll=round(v_nll, 4), val_acc=round(v_acc, 4))), flush=True)
        if best is None or v_nll < best[0]:
            best = (v_nll, step, v_acc, [p.detach().to('cpu', copy=True) for p in trainables])

    for ep in range(epochs):
        order = list(range(len(data)))
        random.Random(seed + ep).shuffle(order)
        for off in range(0, len(order), accum):
            batch = order[off:off + accum]
            frac = steps / max(total - 1, 1)
            scale = (steps + 1) / warmup if steps < warmup else floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * frac))
            for g in opt.param_groups:
                g['lr'] = lr * scale
            opt.zero_grad(set_to_none=True)
            losses = []
            for i in batch:
                x = torch.tensor([data[i]['input_ids']], device='cuda')
                y = torch.tensor([data[i]['labels']], device='cuda')
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss = model(input_ids=x, attention_mask=torch.ones_like(x), labels=y, use_cache=False).loss
                if not torch.isfinite(loss):
                    raise FloatingPointError('nonfinite loss')
                (loss / len(batch)).backward()
                losses.append(float(loss))
            norm = float(torch.nn.utils.clip_grad_norm_(trainables, 1.0))
            opt.step()
            steps += 1
            item = dict(step=steps, epoch=ep + 1, loss=sum(losses) / len(losses), grad_norm=norm, lr=lr * scale, time=time.time())
            journal.write(json.dumps(item) + '\n')
            journal.flush()
            if steps % val_every == 0:
                checkpoint(steps)
    checkpoint(steps)
    journal.close()
    with torch.no_grad():
        for p, b in zip(trainables, best[3]):
            p.data.copy_(b.to(p.device))
    (out / 'best.json').write_text(json.dumps(dict(best_step=best[1], val_nll=best[0], val_acc=best[2], total_steps=steps,
                                                   n_val=len(val_rows), n_train_rows=len(train_rows))))
    vlog.close()
    model.save_pretrained(out / 'adapter', safe_serialization=True)
    return steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=('eval', 'train'))
    ap.add_argument('--stable', action='store_true', help='train_stable: validation slice, warmup+cosine lr, best checkpoint')
    ap.add_argument('--train-source', help="king | opening | all | files joined by '+', e.g. all+dagger1")
    ap.add_argument('--eval-source', default='all')
    ap.add_argument('--adapter', default=None)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--copies', type=int, default=2)
    ap.add_argument('--rel', default='relabel-v2')
    ap.add_argument('--heldout-source', default=None, help='after training, also score this row file (e.g. a teacher-labelled test split)')
    a = ap.parse_args()
    REL_DIR[0] = Path('$AD_ROOT') / a.rel
    out = OUT / a.tag
    train_rows = None
    if a.mode == 'train':
        # before any directory is made or the GPU is touched: every training row must survive the shuffled copy intact
        train_rows = rows_for(a.train_source)
        for r in train_rows:
            check_option_block(r)
    out.mkdir(parents=True, exist_ok=False)
    codec = Codec()
    # 2026-09-23 round 9: the facts version of the training prompts, from the facts-report.json that rerender.py writes
    # into the relabel directory (None for a directory without one, or with a report from before round 9);
    # run_mixed_v4.sh refuses to serve an adapter whose recorded version differs from FACTS_VERSION.
    reg = dict(mode=a.mode, train_source=a.train_source, eval_source=a.eval_source, adapter=a.adapter, epochs=a.epochs, copies=a.copies, rel=a.rel, stable=a.stable,
               facts_version=rel_facts_version(REL_DIR[0]), base=str(BASE), gpu=GPU5090, started=time.time())
    (out / 'registration.json').write_text(json.dumps(reg, indent=1))
    if a.mode == 'eval':
        model = load_model(a.adapter)
        res = evaluate(model, codec, rows_for(a.eval_source), out / 'predictions.jsonl')
    else:
        model = load_model(train=True)
        (out / 'loaded.json').write_text(json.dumps(dict(gpu=torch.cuda.get_device_name(0),
                                                         trainable=sum(p.numel() for p in model.parameters() if p.requires_grad))))
        trainer = train_stable if a.stable else train
        steps = trainer(model, codec, train_rows, out, epochs=a.epochs, copies=a.copies)
        res = dict(optimizer_steps=steps)
        if a.train_source in ('king', 'opening') or a.heldout_source:
            held = a.heldout_source or ('opening' if a.train_source == 'king' else 'king')
            res['heldout_source'] = held
            model.config.use_cache = False
            res['heldout'] = evaluate(model, codec, rows_for(held), out / 'heldout-predictions.jsonl')
    res['finished'] = time.time()
    (out / 'result.json').write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


if __name__ == '__main__':
    main()
