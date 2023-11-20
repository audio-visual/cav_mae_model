# cav_mae_model
 please refer to the original cav_mae repository
 
 How to use it:
 ## Install 
`pip install git+https://github.com/audio-visual/cav_mae_model.git`
 
 ## Get features
 ```python
 import torch
 import cav_mae
 from PIL import Image
 
 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

 model = cav_mae.load('ckpt_path').to(device) 
 
 img_size = 224
 audio_length = 512 # 5 seconds, should match with the audio_length in model checkpoint
 n_mels = 128
 
 # audio_preprocess only used for eval mode,for train mode, you should manually mask it
 visual_preprocess, audio_preprocess = cav_mae.visual_preprocess(img_size), cav_mae.audio_preprocess(audio_length,n_mels)
 
 image = visual_preprocess(Image.open('test.png')).unsqueeze(0).to(device) # 1,3,img_size,img_size
 audio = audio_preprocess('test.wav').unsqueeze(0).to(device) # 1,audio_length,n_mels
 
 # you should manually change the model mode
 model.eval()
 
 # audio_emb: [1,256,768]
 # img_emb: [1,196,768]
 audio_emb, img_emb = model.module.forward_feat(audio, image)
 
 ```

 ## Compute similarity
 ```python
 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

 model = cav_mae.load('ckpt_path').to(device) 
 
 img_size = 224


 visual_preprocess, audio_preprocess = cav_mae.visual_preprocess(img_size), cav_mae.audio_preprocess()

 image = visual_preprocess(Image.open('test.png')).unsqueeze(0).to(device) # 1,3,img_size,img_size
 audio = audio_preprocess('test.wav').unsqueeze(0).to(device) # 1,audio_length,n_mels

 cav_mae.compute_audio_image_similarity(model, audio, image)
 ```

