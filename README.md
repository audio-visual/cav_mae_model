# cav_mae_model
 please refer to the original cav_mae repository
 
 How to use it:
 ```python
 import cav_mae
 from PIL import Image
 model = cav_mae.load('ckpt_path') 
 
 img_size = 224
 audio_length = 512 # 5 seconds, should match with the audio_length in model checkpoint
 n_mels = 128
 
 # audio_preprocess only used for eval mode,for train mode, you should manually mask it
 visual_preprocess, audio_preprocess = cav_mae.visual_preprocess(img_size), cav_mae.audio_preprocess(audio_length,n_mels)
 
 image = visual_preprocess(Image.open('test.png')).unsqueeze(0) # 1,3,img_size,img_size
 audio = audio_preprocess('test.wav').unsqueeze(0) # 1,audio_length,n_mels
 
 # you should manually change the model mode
 model.eval()
 
 audio_emb, img_emb = model.module.forward_feat(audio, image)
 
 ```
