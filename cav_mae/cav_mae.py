import os
import torch 
from typing import Union
import torchvision.transforms as T
from PIL import Image
import PIL
import torchaudio
import numpy as np
from .model import CAVMAE
def build_model(state_dict: dict, multi_gpu=True):
    try:
        _audio_length = int(state_dict['module.pos_embed_a'].size(1)*2) # if the model is trained on multi-gpu
    except:
        _audio_length = int(state_dict['pos_embed_a'].size(1)*2) # if the model is trained on single-gpu
    model = CAVMAE(audio_length=_audio_length,modality_specific_depth=11)
    if multi_gpu:
        if isinstance(model, torch.nn.DataParallel) == False:
            model = torch.nn.DataParallel(model)
            model.load_state_dict(state_dict, strict=False)
    else: # not recommand
        new_state_dict = {}
        for key in state_dict:
            new_key = key.replace('module.','')
            new_state_dict[new_key] = state_dict[key]
        model.load_state_dict(new_state_dict, strict=False)
    
    return model
    
def _convert_image_to_rgb(image):
    return image.convert("RGB")   

def visual_preprocess(img_res):
    return T.Compose([
            T.Resize(img_res, interpolation=PIL.Image.BICUBIC),
            T.CenterCrop(img_res),
            _convert_image_to_rgb,
            T.ToTensor(),
            T.Normalize(
                mean=[0.4850, 0.4560, 0.4060],
                std=[0.2290, 0.2240, 0.2250]
            )]) 

def audio_preprocess(target_length, melbins):
    def _wav2fbank(filename):
        waveform, sr = torchaudio.load(filename)
        waveform = waveform - waveform.mean()
        try:
            fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False, window_type='hanning', num_mel_bins=melbins, dither=0.0, frame_shift=10)
        except:
            fbank = torch.zeros([512, 128]) + 0.01
            print('there is a loading error')

        # target_length = self.target_length
        n_frames = fbank.shape[0]

        p = target_length - n_frames

        # cut and pad
        if p > 0:
            # ReflectionPad2d
            m = torch.nn.ZeroPad2d((0, 0, 0, p)) # TODO 用border填充
            fbank = m(fbank)
        elif p < 0:
            # fbank = fbank[0:target_length, :] # origin

            pt = np.random.randint(-p + 1)
            fbank = fbank[pt : pt + target_length, :]
	fbank = torch.transpose(fbank, 0, 1)	
        return fbank
    return _wav2fbank



def load(name: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", jit: bool = False, download_root: str = None):
    """Load a CLIP model

    Parameters
    ----------
    name : str
        A model name listed by `clip.available_models()`, or the path to a model checkpoint containing the state_dict

    device : Union[str, torch.device]
        The device to put the loaded model

    jit : bool
        Whether to load the optimized JIT model or more hackable non-JIT model (default).

    download_root: str
        path to download the model files; by default, it uses "~/.cache/clip"

    Returns
    -------
    model : torch.nn.Module
        The CLIP model

    preprocess : Callable[[PIL.Image], torch.Tensor]
        A torchvision transform that converts a PIL image into a tensor that the returned model can take as its input
    """
    
    if os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found")

    with open(model_path, 'rb') as opened_file:  
        state_dict = torch.load(opened_file, map_location="cpu")

    
    model = build_model(state_dict).to(device)
    if str(device) == "cpu":
        model.float()
    return model


if __name__ == '__main__':
    pass

