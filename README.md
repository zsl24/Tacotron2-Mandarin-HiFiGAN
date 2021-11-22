# Tacotron2(Mandarin)-HiFiGAN-TTS
Implementation of TTS with combination of Tacotron2 and HiFi-GAN for Mandarin TTS.  

## Inference

In order to inference, we need to download [pre-trained tacotraon2 model](https://github.com/foamliu/Tacotron2-Mandarin/releases/download/v1.0/tacotron2-cn.pt) for mandarin, and place in the root path. Then, we can run `infer_tacotron2_hifigan.py` to get TTS result. We can alter the input text by editting variablle `text` in the `infer_tacotron2_hifigan.py`. Then the result will be saved in the root path named as `output.wav`.  

The pre-trained model of HiFi-GAN has been placed in the `LJ_FT_T2_V3`, which is trained by LJSppech and fine-tuned with Tacotron2. You can find more pre-trained model from [original HiFi-GAN repo](https://github.com/jik876/hifi-gan) with different size and parameters. If you want to try different models or train your own model, please do remember to alter variables in `infer_tacotron2_hifigan.py` to change the path of HiFi-GAN model.  

## Audio Sample
Input: `相对论直接和间接的催生了量子力学的诞生 也为研究微观世界的高速运动确立了全新的数学模型`  
Output: [tacotron2-hifigan.wav](https://github.com/zsl24/Tacotron2-HiFiGAN/blob/main/tacotron2-hifigan.wav)
