import os
os.environ["CUDA_VISIBLE_DEVICES"]= "5"
import librosa
from sklearn.metrics import classification_report
from datasets import load_dataset, load_metric
import torchaudio
import torch
from transformers import AutoConfig, Wav2Vec2Processor, Wav2Vec2FeatureExtractor
from src.models import Wav2Vec2ForSpeechClassification
import numpy as np
import pandas as pd
import torch.nn.functional as F
import json

def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = torchaudio.load(batch["path"])
    speech_array = speech_array.squeeze().numpy()
    #speech_array = librosa.resample(np.asarray(speech_array), sampling_rate, feature_extractor.sampling_rate)
    batch["speech"] = speech_array
    return batch


def predict(batch):
    features = feature_extractor(batch["speech"], sampling_rate=feature_extractor.sampling_rate, return_tensors="pt", padding=True)
    input_values = features.input_values.to(device)

    with torch.no_grad():
        logits = model(input_values).logits 
    scores = F.softmax(logits, dim=1).detach().cpu().numpy()[:,0]
    batch["predicted"] = scores
    return batch

test_dataset = load_dataset("csv", data_files={"test": "dataset/inference_test.csv"}, delimiter="\t")["test"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
print(f"Device: {device}")
model_name_or_path = "good_models/checkpoint-18500-0.89"
print('testing on models ',model_name_or_path)
config = AutoConfig.from_pretrained(model_name_or_path)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name_or_path)
model = Wav2Vec2ForSpeechClassification.from_pretrained(model_name_or_path).to(device)

test_dataset = test_dataset.map(speech_file_to_array_fn)

result = test_dataset.map(predict, batched=True, batch_size=6)

y_pred = result["predicted"]

# Save
outfile = open('inference_results.json','w',encoding='utf-8')
for i in range(len(y_pred)):
    outfile.write(json.dumps(
        {
            "path":result['path'][i],
            "prediction":y_pred[i]
        }
    , ensure_ascii=False)+'\n')
