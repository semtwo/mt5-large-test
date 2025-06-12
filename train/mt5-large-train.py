import torch
from datasets import load_dataset
from transformers import (
    MT5Tokenizer,
    MT5ForConditionalGeneration,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from peft import get_peft_model, LoraConfig, TaskType
import os

# 1. 기본 모델 불러오기
checkpoint_dir = "./checkpoints/mt5-large"
os.makedirs(checkpoint_dir, exist_ok=True)

model_name = "google/mt5-large"
tokenizer = MT5Tokenizer.from_pretrained(model_name)

# 모델 로딩 설정 수정
model = MT5ForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    use_safetensors=True,  # safetensors 사용
    trust_remote_code=True
)

# LoRA 설정
lora_cfg = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q", "v"]
)

model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()  

# 2. 데이터셋 로드
raw_datasets = load_dataset("lcw99/wikipedia-korean-20240501-1million-qna", trust_remote_code=True)
train_data = raw_datasets["train"]

# 학습 데이터의 일부를 검증 데이터로 사용
train_valid_split = train_data.train_test_split(test_size=0.1, seed=42)
train_data = train_valid_split["train"]
valid_data = train_valid_split["test"]

# 3. 데이터 전처리
def make_prompt(example):
    context = example["context"]
    question = example["question"]
    answer = example["answer"]
    
    prompt = (
        f"다음 context를 읽고 질문에 한글로 답하세요. :\n"
        f"context: {context}\n"
        f"input: {question}\n"
        f"output:"
    )
    return {"prompt": prompt, "label": answer}

train_data = train_data.map(make_prompt)
valid_data = valid_data.map(make_prompt)

# 4. 토큰화
#토큰 길이 설정 (파라미터 수정 가능)
max_input_length = 512
max_target_length = 128

def preprocess_function(examples):
    # 인풋: 프롬프트
    inputs = examples["prompt"]
    # 정답: label
    targets = examples["label"]

    model_inputs = tokenizer(
        inputs, 
        text_target=targets,
        max_length=max_input_length, 
        truncation=True,
    )
    return model_inputs

train_dataset = train_data.map(preprocess_function, batched=True, remove_columns=train_data.column_names)
valid_dataset = valid_data.map(preprocess_function, batched=True, remove_columns=valid_data.column_names)



# 5. 학습 설정
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

training_args = Seq2SeqTrainingArguments(
    output_dir=checkpoint_dir,
    num_train_epochs=2,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    # eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=100,
    learning_rate=1e-5,
    max_grad_norm=1.0,
    predict_with_generate=True,
    bf16=True,
    gradient_accumulation_steps=8
)

# 6. 학습 실행
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()

# 7. 모델 저장
model = trainer.model
tokenizer = trainer.tokenizer

new_checkpoint_path = os.path.join(checkpoint_dir, "final_checkpoint")
model.save_pretrained(new_checkpoint_path, safe_serialization=False)
tokenizer.save_pretrained(new_checkpoint_path)