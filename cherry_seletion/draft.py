from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftConfig, PeftModel
import torch
import json
import math
from tqdm import tqdm
import time

# load model and tokenizer
base_model_name = "/mnt/data/nlp.duccd4/repo/LLaMA-Factory/model/Vistral-7B-Chat"
adapter_model_name = "/mnt/data/nlp.duccd4/repo/LLaMA-Factory/saves/Mistral-7B-v0.1/lora/train_2024-07-29-cqc"

model = AutoModelForCausalLM.from_pretrained(base_model_name)
model = PeftModel.from_pretrained(model, adapter_model_name)

tokenizer = AutoTokenizer.from_pretrained(base_model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# sum of pro only answer
def cal_pro_ans(answer, model, tokenizer, device):
    answer_inputs = tokenizer(answer, return_tensors="pt").to(device)
    answer_input_ids = answer_inputs["input_ids"][0]
    N = len(answer_input_ids)
    with torch.no_grad():
        outputs = model(**answer_inputs, labels=answer_inputs["input_ids"])
    log_probs = outputs.logits
    probs = torch.softmax(log_probs, dim=-1)
    
    answer_probs = [probs[0, i, token_id].item() for i, token_id in enumerate(answer_input_ids)]
    answer_probability = 0.0
    for prob in answer_probs:
        answer_probability += math.log(prob)
    return (-1.0 / N) * answer_probability

# sum of pro only answer given context & question
def cal_pro_ans_wque(question, answer, model, tokenizer, device):
    context = f"{question} {answer}"
    context_inputs = tokenizer(context, return_tensors="pt").to(device)
    question_inputs = tokenizer(question, return_tensors="pt").to(device)
    # context = question + answer
    context_input_ids = context_inputs["input_ids"][0]
    question_input_ids = question_inputs["input_ids"][0]
    answer_input_ids = context_input_ids[len(question_input_ids):]
    N = len(answer_input_ids)
    with torch.no_grad():
        outputs = model(**context_inputs, labels=context_inputs["input_ids"])
    log_probs = outputs.logits
    probs = torch.softmax(log_probs, dim=-1)

    answer_probs = [probs[0, len(question_input_ids) + i, token_id].item() for i, token_id in enumerate(answer_input_ids)]
    answer_probability = 0.0
    for prob in answer_probs:
        answer_probability += math.log(prob)
    return (-1.0 / N) * answer_probability

# load data
ifp = "data/remain.json"
with open(ifp, "r", encoding="utf-8") as f:
    remain = json.load(f)
# remain = remain[7:10]

# IFD of 1 sample
def ifd(d):
    ins = d["instruction"]
    inp = d["input"]
    question = ins + "\n" + inp
    answer = d["output"]
    
    numerator = cal_pro_ans_wque(question, answer, model, tokenizer, device)
    denominator = cal_pro_ans(answer, model, tokenizer, device)

    return numerator / denominator

# inference   
start_time = time.time()
i, j = 0, 0
bad_samples = []
scores = []
for d in tqdm(remain, desc="Processing samples"):
    score = ifd(d)
    scores.append(score)
    if score > 1:
        i += 1
        if d["output"] == "":
            j += 1
        d["score"] = score
        bad_samples.append(d)
end_time = time.time()
print(sorted(scores))
print(i, j)
print(f"Duration: {end_time - start_time} seconds")

# save
ofp = "data/bad_samples.json"
with open(ofp, "w", encoding="utf-8") as f:
    json.dump(bad_samples, f, ensure_ascii=False, indent=4)