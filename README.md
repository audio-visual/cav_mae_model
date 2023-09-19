# cav_mae_model
 please refer to the original cav_mae repository
 
 How to use it:
 ```python
 import cav_mae
 model = cav_mae.load('ckpt_path') 
 
 img_size = 224
 audio_length = 512 # 5 seconds, should match with the audio_length in model checkpoint
 n_mels = 128
 audio_preprocess, visual_preprocess = cav_mae.visual_preprocess(img_size), cav_mae.audio_preprocess(audio_length,n_mels)
 
 
 # you should manually change the model mode
 model.eval()
 
 ```
