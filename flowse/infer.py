import os
import torch
import soundfile as sf
import torchaudio

from vocos import Vocos
from loader.datareader import DataReader


import yaml
import time
import argparse
from tqdm import tqdm
from pathlib import Path
import numpy as np

EPS = np.finfo(float).eps
import torch
import librosa
import scipy
from model.model_utils import get_tokenizer
from model import DiT,CFM
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def load_vocoder(vocoder_name="vocos", is_local=False, local_path="", device=None, hf_cache_dir=None):
    if vocoder_name == "vocos":
        # vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device)
        if is_local:
            print(f"Load vocos from local path {local_path}")
            config_path = f"{local_path}/config.yaml"
            model_path = f"{local_path}/pytorch_model.bin"
        else:
            print("Download Vocos from huggingface charactr/vocos-mel-24khz")
            repo_id = "charactr/vocos-mel-24khz"
            config_path = hf_hub_download(repo_id=repo_id, cache_dir=hf_cache_dir, filename="config.yaml")
            model_path = hf_hub_download(repo_id=repo_id, cache_dir=hf_cache_dir, filename="pytorch_model.bin")
        vocoder = Vocos.from_hparams(config_path)
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        from vocos.feature_extractors import EncodecFeatures

        if isinstance(vocoder.feature_extractor, EncodecFeatures):
            encodec_parameters = {
                "feature_extractor.encodec." + key: value
                for key, value in vocoder.feature_extractor.encodec.state_dict().items()
            }
            state_dict.update(encodec_parameters)
        vocoder.load_state_dict(state_dict)
        vocoder = vocoder.eval().to(device)
    elif vocoder_name == "bigvgan":
        try:
            from third_party.BigVGAN import bigvgan
        except ImportError:
            print("You need to follow the README to init submodule and change the BigVGAN source code.")
        if is_local:
            """download from https://huggingface.co/nvidia/bigvgan_v2_24khz_100band_256x/tree/main"""
            vocoder = bigvgan.BigVGAN.from_pretrained(local_path, use_cuda_kernel=False)
        else:
            local_path = snapshot_download(repo_id="nvidia/bigvgan_v2_24khz_100band_256x", cache_dir=hf_cache_dir)
            vocoder = bigvgan.BigVGAN.from_pretrained(local_path, use_cuda_kernel=False)

        vocoder.remove_weight_norm()
        vocoder = vocoder.eval().to(device)
    return vocoder

def normalize(audio, target_level=-25):
    '''Normalize the signal to the target level'''
    rms = (audio ** 2).mean() ** 0.5
    scalar = 10 ** (target_level / 20) / (rms+EPS)
    return scalar * audio


def run(args):
    with open(args.conf, "r") as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    tokenizer = conf['model']['tokenizer']
    tokenizer_path = conf['model']['tokenizer_path']
    conf = conf['infer']
    
    device = torch.device(
        "cuda" if conf["test"]["use_cuda"] and torch.cuda.is_available() else "cpu"
    )

    checkpoint_dir = Path(conf["test"]["checkpoint"])
    cpt_fname = checkpoint_dir / conf["test"]["pt_name"]
    ckpt = torch.load(cpt_fname, map_location=device)
    print("checkpoint: ", cpt_fname)
    print("epoch: ", ckpt["epoch"])
    print("last modified: ", time.ctime(os.path.getmtime(cpt_fname)))
    print("decode wav: ", conf["datareader"]["mix_json"])
    print("save path: ", conf["save"]["dir"])
    
    vocab_char_map, vocab_size = get_tokenizer(tokenizer_path, tokenizer)
    
    model_cls = DiT
    nnet = CFM(transformer=model_cls(**conf['nnet_conf']['arch'], text_num_embeds=vocab_size, mel_dim=conf['nnet_conf']['mel_spec']['n_mel_channels']),
        mel_spec_kwargs=conf['nnet_conf']['mel_spec'],vocab_char_map=vocab_char_map).eval().to(device)
    nnet.load_state_dict(ckpt["model_state_dict"])

    if not os.path.exists(conf["save"]["dir"]):
        os.makedirs(conf["save"]["dir"])
    vocoder_local_path = conf['nnet_conf']['vocoder']['local_path']
    is_local = conf['nnet_conf']['vocoder']['is_local']
    vocoder = load_vocoder(vocoder_name='vocos', is_local=is_local, local_path=vocoder_local_path,device=device)

    Resampler =  torchaudio.transforms.Resample(orig_freq = 16000, new_freq = 24000).to(device)
    data_reader = DataReader(**conf["datareader"])
    
    with torch.no_grad():

        for egs in tqdm(data_reader):
            if egs["utt_id"][-4:] != ".wav":
                egs["utt_id"] = egs["utt_id"] + ".wav"

            mix = egs["mix"].contiguous().to(device)
            text =  egs["text"]
            mix = Resampler(mix)
            
            if conf['test']['cond_type'] == 'noisy':
            
                output,_ = nnet.sample(cond = mix,text=[text])
            elif conf['test']['cond_type'] == 'wotext':
                output,_ = nnet.sample(cond = mix,text=[" "],drop_text=True)


            if egs["utt_id"].find("/") != -1:
                if not os.path.exists(
                    os.path.join(
                        conf["save"]["dir"],
                        egs["utt_id"][
                            egs["utt_id"].find("/") + 1 : egs["utt_id"].rfind("/")
                        ],
                    )
                ):
                    os.makedirs(
                        os.path.join(
                            conf["save"]["dir"],
                            egs["utt_id"][
                                egs["utt_id"].find("/")
                                + 1 : egs["utt_id"].rfind("/")
                            ],
                        )
                    )
                if egs["utt_id"].find("/") == 0:
                    egs["utt_id"] = egs["utt_id"][1:]
                    
            output = output.transpose(-1,-2)
            output = output.to(torch.float32)
            generated_wave = vocoder.decode(output)
            generated_wave = generated_wave.squeeze().cpu().numpy()
            generated_wave = normalize(generated_wave)
            
            generated_wave = librosa.resample(generated_wave,orig_sr=24000,target_sr=16000)
            # quit()
            sf.write(
                os.path.join(conf["save"]["dir"], egs["utt_id"]),
                generated_wave,
                16000,
            )

       


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Command to test model in Pytorch",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-conf", type=str, required=True, help="Yaml configuration file for training"
    )
    args = parser.parse_args()
    run(args)
