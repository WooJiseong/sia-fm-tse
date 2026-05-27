import os
import sys
import random
import itertools
import torch
import torch.nn.functional as F
import torchaudio
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


# ============================================================
# Paths
# ============================================================

FLOWSE_ROOT = os.path.abspath(os.environ.get("FLOWSE_ROOT", os.getcwd()))
PN_REPO = os.environ.get(
    "PN_REPO",
    "/gpfs/home1/eri0529/SIA/paper/TSE-through-Positive-Negative-Enroll",
)

FLOWSE_CKPT = os.environ.get("FLOWSE_CKPT", "wenetspeech4tts_Premium.pt.tar")
PN_CKPT = os.environ.get("PN_CKPT", os.path.join(PN_REPO, "output/proposed-monaural.pt"))

SAVE_DIR = os.environ.get("SAVE_DIR", "output")


# ============================================================
# Import FlowSE modules first
# The FlowSE repo and PN baseline repo both use the top-level package name
# "model". Force Python to resolve "model" from FLOWSE_ROOT here.
# ============================================================

def import_flowse_modules(flowse_root: str, pn_repo: str):
    """Import FlowSE modules robustly across slightly different repo layouts.

    This handles three cases:
    1) official-style `from model import DiT, CFM`
    2) submodule-style `model.cfm.CFM`, `model.backbones.dit.DiT`
    3) local namespace-package clones where classes are located somewhere under
       `model/**/*.py` but are not exported from `model/__init__.py`.
    """
    import importlib
    import pkgutil
    import traceback

    flowse_root = os.path.abspath(flowse_root)
    pn_repo = os.path.abspath(pn_repo)
    model_dir = os.path.join(flowse_root, "model")

    if not os.path.isdir(model_dir):
        raise ImportError(f"FLOWSE_ROOT does not contain a model/ directory: {model_dir}")

    for k in list(sys.modules.keys()):
        if k == "model" or k.startswith("model."):
            del sys.modules[k]

    cleaned_path = []
    for p in sys.path:
        abs_p = os.path.abspath(p or os.getcwd())
        if abs_p == pn_repo:
            continue
        if abs_p not in cleaned_path:
            cleaned_path.append(abs_p)

    sys.path = [flowse_root] + [p for p in cleaned_path if p != flowse_root]

    import_errors = []

    def try_import(module_name):
        try:
            return importlib.import_module(module_name)
        except Exception as exc:
            import_errors.append((module_name, repr(exc)))
            return None

    flowse_model = try_import("model")
    if flowse_model is None:
        raise ImportError(
            f"Failed to import top-level FlowSE model package from {model_dir}. "
            f"First errors: {import_errors[:5]}"
        )

    def find_attr(attr_name, preferred_modules):
        # 1. Already exported from `model`
        if hasattr(flowse_model, attr_name):
            return getattr(flowse_model, attr_name)

        # 2. Common module locations
        for module_name in preferred_modules:
            mod = try_import(module_name)
            if mod is not None and hasattr(mod, attr_name):
                return getattr(mod, attr_name)

        # 3. Last resort: scan all importable modules under model/.
        #    This is intentionally broad because your local FlowSE clone is a
        #    namespace package and does not have model.cfm.
        for module_info in pkgutil.walk_packages([model_dir], prefix="model."):
            module_name = module_info.name
            mod = try_import(module_name)
            if mod is not None and hasattr(mod, attr_name):
                print(f"found FlowSE {attr_name} in {module_name}", flush=True)
                return getattr(mod, attr_name)

        raise ImportError(
            f"Could not find `{attr_name}` anywhere under {model_dir}. "
            f"Run: find {model_dir} -maxdepth 3 -type f -name '*.py' -print"
        )

    DiT = find_attr(
        "DiT",
        [
            "model.backbones.dit",
            "model.dit",
            "model.transformer",
            "model.modules",
            "model.model",
        ],
    )
    CFM = find_attr(
        "CFM",
        [
            "model.cfm",
            "model.flow_matching",
            "model.modules",
            "model.model",
        ],
    )

    # Make namespace-package clones behave like official FlowSE's model package.
    setattr(flowse_model, "DiT", DiT)
    setattr(flowse_model, "CFM", CFM)

    try:
        pn_conditioner = importlib.import_module("model.pn_conditioner")
        model_utils = importlib.import_module("model.model_utils")
    except Exception as exc:
        raise ImportError(
            "Found FlowSE DiT/CFM, but failed to import model.pn_conditioner "
            "or model.model_utils. Make sure pn_conditioner.py is inside "
            f"{model_dir}. Original error: {repr(exc)}"
        ) from exc

    print("FlowSE import resolved:", flush=True)
    print("  FLOWSE_ROOT:", flowse_root, flush=True)
    print("  DiT:", DiT, flush=True)
    print("  CFM:", CFM, flush=True)

    return (
        DiT,
        CFM,
        pn_conditioner.PNConditionedCFM,
        pn_conditioner.PNDiT,
        model_utils.exists,
        model_utils.list_str_to_idx,
        model_utils.list_str_to_tensor,
    )



(
    DiT,
    CFM,
    PNConditionedCFM,
    PNDiT,
    exists,
    list_str_to_idx,
    list_str_to_tensor,
) = import_flowse_modules(FLOWSE_ROOT, PN_REPO)


# ============================================================
# Import PN-TSE modules despite same package name "model"
# ============================================================

def import_pn_modules(pn_repo: str):
    import importlib

    pn_repo = os.path.abspath(pn_repo)
    flowse_root = os.path.abspath(FLOWSE_ROOT)

    saved_modules = {
        k: v for k, v in sys.modules.items()
        if k == "model" or k.startswith("model.") or k == "dataset" or k.startswith("dataset.")
    }

    for k in list(sys.modules.keys()):
        if k == "model" or k.startswith("model.") or k == "dataset" or k.startswith("dataset."):
            del sys.modules[k]

    saved_path = list(sys.path)

    sys.path = [
        p for p in sys.path
        if os.path.abspath(p or os.getcwd()) != flowse_root
    ]

    # Put PN repo first so "model.*" and "dataset.*" resolve to PN-Enroll.
    sys.path.insert(0, pn_repo)

    try:
        LibriDataset_single_emb = importlib.import_module(
            "dataset.LibriSpeech_single_emb"
        ).LibriDataset_single_emb

        TFGridNet_encoder = importlib.import_module(
            "model.tfgridnet_encoder"
        ).TFGridNet_encoder

        GridNetBlock_attnhead = importlib.import_module(
            "model.GridnetAttnHead"
        ).GridNetBlock_attnhead

        TFGridNet_KVfusion = importlib.import_module(
            "model.tfgridnet_KVfusion"
        ).TFGridNet_KVfusion

        TFGridNet_origcrossattn_causal_single_emb = importlib.import_module(
            "model.tfgridnet_crossattn_causal_single_emb"
        ).TFGridNet_origcrossattn_causal_single_emb

    finally:
        for k in list(sys.modules.keys()):
            if k == "model" or k.startswith("model.") or k == "dataset" or k.startswith("dataset."):
                del sys.modules[k]

        sys.path = saved_path
        sys.modules.update(saved_modules)

    return (
        LibriDataset_single_emb,
        TFGridNet_encoder,
        GridNetBlock_attnhead,
        TFGridNet_KVfusion,
        TFGridNet_origcrossattn_causal_single_emb,
    )



(
    LibriDataset_single_emb,
    TFGridNet_encoder,
    GridNetBlock_attnhead,
    TFGridNet_KVfusion,
    TFGridNet_origcrossattn_causal_single_emb,
) = import_pn_modules(PN_REPO)


# ============================================================
# Config
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)
print("PN_REPO:", PN_REPO)
print("FLOWSE_CKPT:", FLOWSE_CKPT)
print("PN_CKPT:", PN_CKPT)

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "1"))
NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", "100"))
STEPS_PER_EPOCH = int(os.environ.get("STEPS_PER_EPOCH", "2000"))
LR = float(os.environ.get("LR", "1e-4"))
FLOWSE_LR = float(os.environ.get("FLOWSE_LR", "2e-7"))
TEMPORAL_POOL = os.environ.get("TEMPORAL_POOL", "mean_std")
PN_DROPOUT = float(os.environ.get("PN_DROPOUT", "0.0"))
TOKEN_MODE = os.environ.get("TOKEN_MODE", "full_2d")
INJECTION_MODE = os.environ.get("INJECTION_MODE", "block")
MAX_FULL_TOKENS_ENV = os.environ.get("MAX_FULL_TOKENS", "")
MAX_FULL_TOKENS = int(MAX_FULL_TOKENS_ENV) if MAX_FULL_TOKENS_ENV else None
UNFREEZE_FLOWSE_MODE = os.environ.get("UNFREEZE_FLOWSE_MODE", "all")
UNFREEZE_LAST_DIT_BLOCKS = int(os.environ.get("UNFREEZE_LAST_DIT_BLOCKS", "8"))
CLEAN_EST_L1_WEIGHT = float(os.environ.get("CLEAN_EST_L1_WEIGHT", "1.0"))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "1e-4"))
WARMUP_STEPS = int(os.environ.get("WARMUP_STEPS", "2000"))
MIN_LR_RATIO = float(os.environ.get("MIN_LR_RATIO", "0.1"))
GRAD_CLIP = float(os.environ.get("GRAD_CLIP", "1.0"))

# Long-run training utilities
RESUME_CKPT = os.environ.get("RESUME_CKPT", "")
RESUME_OPTIMIZER = int(os.environ.get("RESUME_OPTIMIZER", "1"))
BEST_CKPT_NAME = os.environ.get("BEST_CKPT_NAME", "best.pt")
LOG_DIR = os.environ.get("LOG_DIR", os.path.join(SAVE_DIR, "tensorboard"))
LOG_INTERVAL = int(os.environ.get("LOG_INTERVAL", "20"))
SAVE_EVERY_EPOCH = int(os.environ.get("SAVE_EVERY_EPOCH", "1"))

print("BATCH_SIZE:", BATCH_SIZE)
print("NUM_EPOCHS:", NUM_EPOCHS)
print("STEPS_PER_EPOCH:", STEPS_PER_EPOCH)
print("LR:", LR)
print("FLOWSE_LR:", FLOWSE_LR)
print("TEMPORAL_POOL:", TEMPORAL_POOL)
print("PN_DROPOUT:", PN_DROPOUT)
print("TOKEN_MODE:", TOKEN_MODE)
print("INJECTION_MODE:", INJECTION_MODE)
print("MAX_FULL_TOKENS:", MAX_FULL_TOKENS)
print("UNFREEZE_FLOWSE_MODE:", UNFREEZE_FLOWSE_MODE)
print("UNFREEZE_LAST_DIT_BLOCKS:", UNFREEZE_LAST_DIT_BLOCKS)
print("CLEAN_EST_L1_WEIGHT:", CLEAN_EST_L1_WEIGHT)
print("WEIGHT_DECAY:", WEIGHT_DECAY)
print("WARMUP_STEPS:", WARMUP_STEPS)
print("MIN_LR_RATIO:", MIN_LR_RATIO)
print("GRAD_CLIP:", GRAD_CLIP)
print("RESUME_CKPT:", RESUME_CKPT)
print("RESUME_OPTIMIZER:", RESUME_OPTIMIZER)
print("BEST_CKPT_NAME:", BEST_CKPT_NAME)
print("LOG_DIR:", LOG_DIR)
print("LOG_INTERVAL:", LOG_INTERVAL)
print("SAVE_EVERY_EPOCH:", SAVE_EVERY_EPOCH)

sample_rate = 16000
source_num = 3
enroll_num = 3
active_num = [-1, 1, -1]
wave_length = 3
pos_example_length = 48000
neg_example_length = 48000
snr_db_range = [0, 0]
filling_pattern = "repeat"
dvec_rate = 50
reverb_cond = False

pn_data_dir = os.environ.get(
    "PN_DATA_DIR",
    os.path.join(PN_REPO, "data/LibriSpeech/LibriSpeech/_test_data/"),
)
pn_noise_dir = os.environ.get(
    "PN_NOISE_DIR",
    os.path.join(PN_REPO, "data/wham_noise/tt/"),
)
print("PN_DATA_DIR:", pn_data_dir)
print("PN_NOISE_DIR:", pn_noise_dir)


# ============================================================
# Build frozen PN-TSE encoder model
# ============================================================

encoder = TFGridNet_encoder(
    num_ch=2,
    n_fft=128,
    stride=64,
    num_blocks=3,
    binaural=False,
)

encoder_head = GridNetBlock_attnhead(
    layer_num=2,
    pooling_size=1,
    stride=1,
)

pn_model = TFGridNet_origcrossattn_causal_single_emb(
    n_fft=128,
    stride=64,
    n_layers=3,
    lstm_hidden_units=64,
    emb_dim=64,
    emb_ks=1,
    model_normalize=True,
    Fusion_class=TFGridNet_KVfusion,
    pooling_size=40,
    fusion_stride=40,
    encoder=encoder,
    encoder_head=encoder_head,
    train_encoder=False,
    train_encoder_head=False,
    fusion_layer=[0, 1],
    binaural=False,
).to(device)

pn_ckpt = torch.load(PN_CKPT, map_location="cpu")
pn_state = pn_ckpt["state_dict"]
model_state = pn_model.state_dict()

compatible_state = {}
skipped = []

for k, v in pn_state.items():
    if k in model_state and model_state[k].shape == v.shape:
        compatible_state[k] = v
    else:
        skipped.append((k, tuple(v.shape), tuple(model_state[k].shape) if k in model_state else None))

missing, unexpected = pn_model.load_state_dict(compatible_state, strict=False)

print(f"loaded compatible PN-TSE weights: {len(compatible_state)}")
print(f"skipped incompatible PN-TSE weights: {len(skipped)}")
for item in skipped[:20]:
    print("  skipped:", item)
print("missing keys:", len(missing))
print("unexpected keys:", len(unexpected))

pn_model.eval()

for p in pn_model.parameters():
    p.requires_grad = False

print("loaded frozen PN-TSE encoder")


# ============================================================
# Build online PN dataset
# reproducable=False follows the training-style random sampling behavior.
# ============================================================

dataset = LibriDataset_single_emb(
    pn_data_dir,
    sample_rate=sample_rate,
    wave_length=wave_length * sample_rate,
    pos_example_length=pos_example_length,
    neg_example_length=neg_example_length,
    snr_db_range=snr_db_range,
    source_num=source_num,
    min_source_num=source_num,
    enroll_num=enroll_num,
    min_enroll_num=enroll_num,
    active_num=active_num,
    reproducable=False,
    normalize=False,
    filling_pattern=filling_pattern,
    return_dvec=False,
    dvec_rate=dvec_rate,
    include_silent=False,
    special_spk=[],
    reverb="none",
    binaural=False,
    reverb_cond=reverb_cond,
    zero_in_tgt=False,
    noise_dir=pn_noise_dir,
    same_disturb=False,
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    drop_last=True,
)

loader_iter = itertools.cycle(loader)

print("online dataset length:", len(dataset))
print("batch_size:", BATCH_SIZE)
print("epochs:", NUM_EPOCHS)
print("steps_per_epoch:", STEPS_PER_EPOCH)
print("lr:", LR)


# ============================================================
# Build pretrained FlowSE + PN adapter
# ============================================================

flowse_ckpt = torch.load(FLOWSE_CKPT, map_location="cpu")
flowse_state = flowse_ckpt["model_state_dict"]

with open("config/train.yaml", "r") as f:
    conf = yaml.safe_load(f)

model_conf = conf["model"]
arch = dict(model_conf["arch"])
mel_conf = dict(model_conf["mel_spec"])

text_num_embeds = flowse_state["transformer.text_embed.text_embed.weight"].shape[0] - 1
mel_dim = mel_conf["n_mel_channels"]
target_sr = mel_conf["target_sample_rate"]

transformer = DiT(
    **arch,
    text_num_embeds=text_num_embeds,
    mel_dim=mel_dim,
)

cfm = CFM(
    transformer=transformer,
    audio_drop_prob=model_conf.get("audio_drop_prob", 0.0),
    cond_drop_prob=model_conf.get("cond_drop_prob", 0.0),
    num_channels=mel_dim,
    mel_spec_kwargs=mel_conf,
)

cfm.load_state_dict(flowse_state, strict=True)

cfm.transformer = PNDiT(
    cfm.transformer,
    spk_dim=64,
    freq_bins=65,
    temporal_pool=TEMPORAL_POOL,
    dropout=PN_DROPOUT,
    token_mode=TOKEN_MODE,
    max_full_tokens=MAX_FULL_TOKENS,
    injection_mode=INJECTION_MODE,
)

cfm = cfm.to(device)
cfm.eval()

for p in cfm.parameters():
    p.requires_grad = False

for p in cfm.transformer.pn_projector.parameters():
    p.requires_grad = True

for p in cfm.transformer.speaker_attn.parameters():
    p.requires_grad = True

for p in cfm.transformer.query_proj.parameters():
    p.requires_grad = True

for p in cfm.transformer.out_proj.parameters():
    p.requires_grad = True

cfm.transformer.speaker_attn_gate.requires_grad = True

model = PNConditionedCFM(cfm=cfm).to(device)

def set_requires_grad(module, requires_grad):
    for param in module.parameters():
        param.requires_grad = requires_grad


def unfreeze_flowse(pn_dit, mode, num_blocks):
    mode = mode.lower()
    if mode not in {"none", "tail", "all"}:
        raise ValueError("UNFREEZE_FLOWSE_MODE must be one of: none, tail, all")

    if mode == "none":
        return []

    base = pn_dit.base_dit
    trainable = []

    if mode == "all":
        set_requires_grad(base, True)
        base.train()
        return list(base.parameters())

    if num_blocks <= 0:
        return []

    for block in base.transformer_blocks[-num_blocks:]:
        set_requires_grad(block, True)
        block.train()
        trainable.extend(block.parameters())

    # Let the pretrained model adapt its condition interpretation and output head.
    for module_name in ("input_embed", "norm_out", "proj_out"):
        module = getattr(base, module_name, None)
        if module is not None:
            set_requires_grad(module, True)
            module.train()
            trainable.extend(module.parameters())

    return trainable


flowse_params = unfreeze_flowse(
    cfm.transformer,
    UNFREEZE_FLOWSE_MODE,
    UNFREEZE_LAST_DIT_BLOCKS,
)
adapter_params = [
    p
    for name, p in model.named_parameters()
    if p.requires_grad and not name.startswith("cfm.transformer.base_dit.")
]

param_groups = [{"params": adapter_params, "lr": LR, "weight_decay": WEIGHT_DECAY}]
if flowse_params:
    param_groups.append({"params": flowse_params, "lr": FLOWSE_LR, "weight_decay": WEIGHT_DECAY})

trainable_params = [p for group in param_groups for p in group["params"]]
optimizer = torch.optim.AdamW(param_groups)


def lr_lambda(step):
    if WARMUP_STEPS > 0 and step < WARMUP_STEPS:
        return max((step + 1) / WARMUP_STEPS, MIN_LR_RATIO)
    total_steps = max(NUM_EPOCHS * STEPS_PER_EPOCH, 1)
    decay_steps = max(total_steps - WARMUP_STEPS, 1)
    progress = min(max((step - WARMUP_STEPS) / decay_steps, 0.0), 1.0)
    cosine = 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.141592653589793))).item()
    return MIN_LR_RATIO + (1.0 - MIN_LR_RATIO) * cosine


scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

print("loaded pretrained FlowSE")
print("checkpoint epoch:", flowse_ckpt.get("epoch"))
print("checkpoint best_loss:", flowse_ckpt.get("best_loss"))
print("trainable params:", sum(p.numel() for p in trainable_params))
print("adapter trainable params:", sum(p.numel() for p in adapter_params))
print("flowse trainable params:", sum(p.numel() for p in flowse_params))


def tokenize_text(cfm_module, text, batch, device):
    if isinstance(text, list):
        if exists(cfm_module.vocab_char_map):
            text = list_str_to_idx(text, cfm_module.vocab_char_map).to(device)
        else:
            text = list_str_to_tensor(text).to(device)
        assert text.shape[0] == batch
    return text


def pn_flow_matching_step(cfm_module, inp, clean, text, cond_emb):
    """CFM training step with a clean-mel reconstruction proxy.

    The PN condition is injected into the DiT vector field, not added to the
    noisy mel condition. clean_est = x_t + (1 - t) * v_theta(x_t, t).
    """
    if inp.ndim == 2:
        inp = cfm_module.mel_spec(inp).permute(0, 2, 1)
        clean = cfm_module.mel_spec(clean).permute(0, 2, 1)
        assert inp.shape[-1] == cfm_module.num_channels

    batch = inp.shape[0]
    dtype = inp.dtype
    device = cfm_module.device
    text = tokenize_text(cfm_module, text, batch, device)

    x1 = clean
    x0 = torch.randn_like(x1)
    time = torch.rand((batch,), dtype=dtype, device=device)
    t = time.unsqueeze(-1).unsqueeze(-1)
    xt = (1 - t) * x0 + t * x1
    flow = x1 - x0

    cond = inp
    drop_audio_cond = random.random() < cfm_module.audio_drop_prob
    if random.random() < cfm_module.cond_drop_prob:
        drop_audio_cond = True
        drop_text = True
    else:
        drop_text = False

    cfm_module.transformer.set_condition(cond_emb)
    try:
        pred = cfm_module.transformer(
            x=xt,
            cond=cond,
            text=text,
            time=time,
            drop_audio_cond=drop_audio_cond,
            drop_text=drop_text,
        )
    finally:
        cfm_module.transformer.clear_condition()

    cfm_loss = F.mse_loss(pred, flow, reduction="mean")
    clean_est = xt + (1 - t) * pred
    clean_est_l1 = F.l1_loss(clean_est, x1)
    return cfm_loss, cond, pred, clean_est, clean_est_l1


# ============================================================
# Train
# ============================================================

os.makedirs(SAVE_DIR, exist_ok=True)
writer = SummaryWriter(LOG_DIR)
print("TensorBoard log dir:", LOG_DIR, flush=True)


def make_checkpoint(epoch, global_step, epoch_loss, epoch_cfm_loss, epoch_clean_est_l1, best_loss):
    return {
        "epoch": epoch,
        "global_step": global_step,
        "pn_projector": model.cfm.transformer.pn_projector.state_dict(),
        "speaker_attn": model.cfm.transformer.speaker_attn.state_dict(),
        "query_proj": model.cfm.transformer.query_proj.state_dict(),
        "out_proj": model.cfm.transformer.out_proj.state_dict(),
        "speaker_attn_gate": model.cfm.transformer.speaker_attn_gate.detach().cpu(),
        "epoch_loss": epoch_loss,
        "epoch_cfm_loss": epoch_cfm_loss,
        "epoch_clean_est_l1": epoch_clean_est_l1,
        "best_loss": best_loss,
        "temporal_pool": TEMPORAL_POOL,
        "token_mode": TOKEN_MODE,
        "injection_mode": INJECTION_MODE,
        "max_full_tokens": MAX_FULL_TOKENS,
        "clean_est_l1_weight": CLEAN_EST_L1_WEIGHT,
        "flowse_tail": {
            name: param.detach().cpu()
            for name, param in model.cfm.transformer.base_dit.named_parameters()
            if param.requires_grad
        },
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "unfreeze_flowse_mode": UNFREEZE_FLOWSE_MODE,
        "unfreeze_last_dit_blocks": UNFREEZE_LAST_DIT_BLOCKS,
        "flowse_lr": FLOWSE_LR,
        "weight_decay": WEIGHT_DECAY,
        "warmup_steps": WARMUP_STEPS,
        "min_lr_ratio": MIN_LR_RATIO,
        "batch_size": BATCH_SIZE,
        "steps_per_epoch": STEPS_PER_EPOCH,
        "lr": LR,
        "online_pn_encoder": True,
        "pn_repo": PN_REPO,
    }


def move_optimizer_state_to_device(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


start_epoch = 0
global_step = 0
best_loss = float("inf")

if RESUME_CKPT:
    print("resuming from:", RESUME_CKPT, flush=True)
    resume = torch.load(RESUME_CKPT, map_location="cpu")

    model.cfm.transformer.pn_projector.load_state_dict(resume["pn_projector"])
    model.cfm.transformer.speaker_attn.load_state_dict(resume["speaker_attn"])
    model.cfm.transformer.query_proj.load_state_dict(resume["query_proj"])
    model.cfm.transformer.out_proj.load_state_dict(resume["out_proj"])

    if "speaker_attn_gate" in resume:
        model.cfm.transformer.speaker_attn_gate.data.copy_(
            resume["speaker_attn_gate"].to(device)
        )

    flowse_tail = resume.get("flowse_tail", {})
    if flowse_tail:
        base_state = model.cfm.transformer.base_dit.state_dict()
        base_state.update(flowse_tail)
        model.cfm.transformer.base_dit.load_state_dict(base_state, strict=True)
        print("loaded resumed flowse_tail params:", len(flowse_tail), flush=True)

    if RESUME_OPTIMIZER and "optimizer" in resume:
        try:
            optimizer.load_state_dict(resume["optimizer"])
            move_optimizer_state_to_device(optimizer, device)
            print("loaded optimizer state", flush=True)
        except ValueError as exc:
            print(f"skipped optimizer state because parameter groups changed: {exc}", flush=True)

    if RESUME_OPTIMIZER and "scheduler" in resume:
        try:
            scheduler.load_state_dict(resume["scheduler"])
            print("loaded scheduler state", flush=True)
        except Exception as exc:
            print(f"skipped scheduler state: {exc}", flush=True)

    start_epoch = int(resume.get("epoch", 0))
    global_step = int(resume.get("global_step", 0))
    best_loss = float(resume.get("best_loss", resume.get("epoch_loss", float("inf"))))

    print(
        f"resume done | start_epoch={start_epoch} | global_step={global_step} | best_loss={best_loss}",
        flush=True,
    )

model.train()
cfm.eval()
cfm.transformer.pn_projector.train()
cfm.transformer.speaker_attn.train()
cfm.transformer.query_proj.train()
cfm.transformer.out_proj.train()
if flowse_params:
    cfm.transformer.base_dit.train()

try:
    for epoch in range(start_epoch, NUM_EPOCHS):
        total_loss = 0.0
        total_cfm_loss = 0.0
        total_clean_est_l1 = 0.0
        count = 0

        for step in range(STEPS_PER_EPOCH):
            global_step += 1

            audio, pos, neg = next(loader_iter)

            audio = audio.to(device)  # [B, source/noise, ch, wav]
            pos = pos.to(device)
            neg = neg.to(device)

            mix_wave = audio.sum(dim=1).squeeze(1)  # [B, wav]
            target_wave = audio[:, :active_num[1]].sum(dim=1).squeeze(1)  # [B, wav]

            with torch.no_grad():
                # Dataset already gives 3s pos/neg according to pos_example_length/neg_example_length.
                # Because reproducable=False, repeated calls generate different enrollment examples.
                cond_emb = pn_model.encode(
                    pos.sum(dim=1),
                    neg.sum(dim=1),
                )  # [B, 64, T, 65]

                if sample_rate != target_sr:
                    resampler = torchaudio.transforms.Resample(
                        orig_freq=sample_rate,
                        new_freq=target_sr,
                    ).to(device)
                    mix_wave = resampler(mix_wave)
                    target_wave = resampler(target_wave)

                noisy_mel = cfm.mel_spec(mix_wave).permute(0, 2, 1)
                clean_mel = cfm.mel_spec(target_wave).permute(0, 2, 1)

            text = [" "] * noisy_mel.shape[0]

            optimizer.zero_grad()

            cfm_loss, cond, pred, clean_est, clean_est_l1 = pn_flow_matching_step(
                cfm,
                inp=noisy_mel,
                clean=clean_mel,
                text=text,
                cond_emb=cond_emb,
            )
            loss = cfm_loss + CLEAN_EST_L1_WEIGHT * clean_est_l1

            if not torch.isfinite(loss):
                print(f"non-finite loss at epoch {epoch+1}, step {step+1}, global_step {global_step}: {loss.item()}")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, GRAD_CLIP)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            total_cfm_loss += cfm_loss.item()
            total_clean_est_l1 += clean_est_l1.item()
            count += 1

            if global_step == 1 or global_step % LOG_INTERVAL == 0:
                with torch.no_grad():
                    pn_tokens = model.cfm.transformer.pn_projector(cond_emb)
                    pn_norm = pn_tokens.norm(dim=-1).mean().item()
                    gate = model.cfm.transformer.speaker_attn_gate.item()
                    clean_est_delta = (clean_est - clean_mel).abs().mean().item()
                    current_lrs = [group["lr"] for group in optimizer.param_groups]

                print(
                    f"epoch {epoch+1:03d} | step {global_step:06d} | "
                    f"loss {loss.item():.6f} | cfm {cfm_loss.item():.6f} | "
                    f"clean_l1 {clean_est_l1.item():.6f} | avg_loss {total_loss / max(count, 1):.6f} | "
                    f"pn_token_norm {pn_norm:.6f} | gate {gate:.6f} | clean_delta {clean_est_delta:.6f} | "
                    f"lr {current_lrs}",
                    flush=True,
                )

                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/cfm_loss", cfm_loss.item(), global_step)
                writer.add_scalar("train/clean_est_l1", clean_est_l1.item(), global_step)
                writer.add_scalar("train/pn_token_norm", pn_norm, global_step)
                writer.add_scalar("train/speaker_attn_gate", gate, global_step)
                writer.add_scalar("train/clean_delta", clean_est_delta, global_step)
                for group_idx, lr_value in enumerate(current_lrs):
                    writer.add_scalar(f"lr/group_{group_idx}", lr_value, global_step)

        epoch_loss = total_loss / max(count, 1)
        epoch_cfm_loss = total_cfm_loss / max(count, 1)
        epoch_clean_est_l1 = total_clean_est_l1 / max(count, 1)
        print(
            f"epoch {epoch+1:03d} done | epoch_loss {epoch_loss:.6f} | "
            f"cfm {epoch_cfm_loss:.6f} | clean_l1 {epoch_clean_est_l1:.6f}",
            flush=True,
        )

        is_best = epoch_loss < best_loss
        if is_best:
            best_loss = epoch_loss

        writer.add_scalar("epoch/loss", epoch_loss, epoch + 1)
        writer.add_scalar("epoch/cfm_loss", epoch_cfm_loss, epoch + 1)
        writer.add_scalar("epoch/clean_est_l1", epoch_clean_est_l1, epoch + 1)
        writer.add_scalar("epoch/best_loss", best_loss, epoch + 1)
        writer.flush()

        checkpoint = make_checkpoint(
            epoch=epoch + 1,
            global_step=global_step,
            epoch_loss=epoch_loss,
            epoch_cfm_loss=epoch_cfm_loss,
            epoch_clean_est_l1=epoch_clean_est_l1,
            best_loss=best_loss,
        )

        if SAVE_EVERY_EPOCH > 0 and ((epoch + 1) % SAVE_EVERY_EPOCH == 0):
            save_path = os.path.join(
                SAVE_DIR,
                f"pn_online_adapter_bs{BATCH_SIZE}_epoch{epoch+1}.pt",
            )
            torch.save(checkpoint, save_path)
            print("saved:", save_path, flush=True)

        if is_best:
            best_path = os.path.join(SAVE_DIR, BEST_CKPT_NAME)
            torch.save(checkpoint, best_path)
            print(f"saved best checkpoint: {best_path} | best_loss {best_loss:.6f}", flush=True)

finally:
    writer.close()

print("online PN-conditioned FlowSE adapter training done")