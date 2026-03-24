"""Diagnose where the Sec-LogLLM inference pipeline is failing.

Test 1: Bypass alpha gate (set alpha=1) — tests if projector+decoder work
Test 2: Use Stage 1 decoder with raw text prompt (no encoder at all)
Test 3: Inspect actual intermediate values (encoder output, gate logits, etc.)
"""
from __future__ import annotations
import json, sys, torch
from pathlib import Path
from peft import PeftModel
from torch.serialization import safe_globals
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.config import ExperimentConfig
from src.models.encoder import EncoderConfig
from src.models.projector import ProjectorConfig
from src.models.sec_logllm import SecLogLLM, SecLogLLMConfig

STAGE1 = ROOT / "results/checkpoints/stage1_final/final_adapter"
STAGE3 = ROOT / "results/checkpoints/stage3/stage3_epoch_1.pt"
INPUT  = ROOT / "data/processed/val_labeled_strat10k_seed42.jsonl"

# Pick 5 samples: mix of attack and normal
SAMPLES = []
with open(INPUT) as f:
    for line in f:
        d = json.loads(line)
        SAMPLES.append(d)
        if len(SAMPLES) >= 200:
            break
# Get some attacks
attacks = [s for s in SAMPLES if s.get("label") == "Attack"][:3]
normals = [s for s in SAMPLES if s.get("label") == "Normal"][:2]
TEST_LOGS = attacks + normals
if not attacks:
    TEST_LOGS = SAMPLES[:5]
print(f"Test samples: {len(TEST_LOGS)} ({len(attacks)} attacks, {len(normals)} normals)")
for i, s in enumerate(TEST_LOGS):
    print(f"  [{i}] label={s.get('label')}, raw={s['raw'][:80]}...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────
# TEST 1: Raw text prompt with Stage 1 LoRA (no encoder)
# ─────────────────────────────────────────────────────
print("\n" + "="*60)
print("TEST 1: Stage 1 decoder + raw text prompt (no encoder/projector)")
print("="*60)

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct", trust_remote_code=True)
peft_model = PeftModel.from_pretrained(base, STAGE1)
peft_model.eval().to(device)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")

for i, sample in enumerate(TEST_LOGS):
    prompt = (
        "<|im_start|>system\n"
        "You are a security analyst specializing in log analysis and threat detection.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"Analyze this log for security threats:\n{sample['raw']}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    inputs = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = peft_model.generate(**inputs, max_new_tokens=200, do_sample=False)
    response = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"\n[{i}] GT={sample.get('label')} | Response: {response[:200]}")

del peft_model, base
torch.cuda.empty_cache()

# ─────────────────────────────────────────────────────
# TEST 2: Inspect intermediate values in the full pipeline
# ─────────────────────────────────────────────────────
print("\n" + "="*60)
print("TEST 2: Full pipeline — inspecting intermediate values")
print("="*60)

config = ExperimentConfig()
model = SecLogLLM(config.sec_logllm)

base2 = AutoModelForCausalLM.from_pretrained(config.sec_logllm.decoder_model_name_or_path, trust_remote_code=True)
peft2 = PeftModel.from_pretrained(base2, STAGE1)
peft2.eval()
model.decoder_model = peft2

decoder_dtype = next(model.decoder_model.parameters()).dtype
with safe_globals([SecLogLLMConfig, EncoderConfig, ProjectorConfig]):
    ckpt = torch.load(STAGE3, map_location="cpu")
model.encoder.load_state_dict(ckpt["encoder_state"])
model.projector.load_state_dict(ckpt["projector_state"])
if "decoder_lora_state" in ckpt:
    model.decoder_model.load_state_dict(ckpt["decoder_lora_state"], strict=False)
model.encoder.to(dtype=decoder_dtype)
model.projector.to(dtype=decoder_dtype)
model.to(device)

for i, sample in enumerate(TEST_LOGS):
    log_text = sample["raw"]
    
    # Step-by-step inspection
    with torch.no_grad():
        # 1. Encode
        tokenized = model.encoder.tokenize([log_text], device=device)
        enc_out = model.encoder(
            input_ids=tokenized["input_ids"],
            attention_mask=tokenized["attention_mask"],
            token_type_ids=tokenized.get("token_type_ids"),
        )
        enc_emb = enc_out.pooled_embeddings  # (1, 768)
        
        # 2. Gate logits (before sigmoid)
        gate_logits = model.projector.gate(enc_emb)
        alpha_raw = torch.sigmoid(gate_logits)
        
        # 3. Projected
        projected = model.projector.projection(enc_emb)
        gated = projected * alpha_raw
        
    print(f"\n[{i}] GT={sample.get('label')}")
    print(f"  Encoder output: mean={enc_emb.float().mean():.4f}, std={enc_emb.float().std():.4f}, "
          f"min={enc_emb.float().min():.4f}, max={enc_emb.float().max():.4f}, norm={enc_emb.float().norm():.4f}")
    print(f"  Gate logit: {gate_logits.item():.4f}")
    print(f"  Alpha (sigmoid): {alpha_raw.item():.10f}")
    print(f"  Projected norm: {projected.float().norm():.4f}")
    print(f"  Gated norm: {gated.float().norm():.4f}")

# ─────────────────────────────────────────────────────
# TEST 3: Bypass alpha — force alpha=1
# ─────────────────────────────────────────────────────
print("\n" + "="*60)
print("TEST 3: Bypass alpha gate (force alpha=1)")
print("="*60)

# Monkey-patch the projector to bypass gating
original_forward = model.projector.forward
def bypass_gate(embeddings):
    projected = model.projector.projection(embeddings)
    if model.projector.layernorm:
        projected = model.projector.layernorm(projected)
    if model.projector.dropout:
        projected = model.projector.dropout(projected)
    alphas = torch.ones(embeddings.size(0), 1, device=embeddings.device, dtype=embeddings.dtype)
    return projected, alphas

model.projector.forward = bypass_gate

STRICT_JSON = (
    "Respond with a single JSON object only. Do not output markdown, lists, or extra text. "
    "Use keys exactly: label, mitre_t_code, reasoning. "
    "label must be Attack or Normal; mitre_t_code can be null.\n"
    "<|im_end|>\n"
    "<|im_start|>assistant\n"
)

for i, sample in enumerate(TEST_LOGS):
    texts, alphas = model.generate(
        [[sample["raw"]]],
        prompt_suffixes=STRICT_JSON,
        max_new_tokens=200,
        do_sample=False,
    )
    print(f"\n[{i}] GT={sample.get('label')} | Alpha={alphas[0].item():.4f}")
    print(f"  Response: {texts[0][:200]}")

model.projector.forward = original_forward
print("\n" + "="*60)
print("DIAGNOSIS COMPLETE")
print("="*60)
