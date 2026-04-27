import json
import torch
import os
from huggingface_hub import login
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# ── 설정 ──────────────────────────────────────────────────────
HF_TOKEN   = "hf_여기에_토큰_입력"   # HuggingFace Access Token
MODEL_ID   = "meta-llama/Llama-3.2-1B-Instruct"
TRAIN_JSON = "dataset/llama_train_data_cleaned.json"
VAL_JSON   = "dataset/llama_val_data_cleaned.json"
SAVE_PATH  = "./output"

MAX_LENGTH = 1024
BATCH_SIZE = 32       # A100 80GB 기준
GRAD_ACCUM = 2        # 실질 배치 = 64
EPOCHS     = 3
LR         = 2e-4
VAL_SPLIT  = 0.05     # val.json 없을 때 train에서 5% 분리
# ──────────────────────────────────────────────────────────────


def load_data():
    with open(TRAIN_JSON, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    if os.path.exists(VAL_JSON):
        with open(VAL_JSON, "r", encoding="utf-8") as f:
            val_data = json.load(f)
    else:
        split = int(len(train_data) * (1 - VAL_SPLIT))
        val_data   = train_data[split:]
        train_data = train_data[:split]
        print(f"val.json 없음 — train에서 {len(val_data):,}개 분리")

    return Dataset.from_list(train_data), Dataset.from_list(val_data)


def format_prompt(row):
    instruction = row.get("instruction", "당신은 산악 응급처치 전문 AI입니다.")
    input_text  = row.get("input", "")
    output_text = row.get("output", "")

    return {
        "text": (
            f"<|begin_of_text|>"
            f"<|start_header_id|>system<|end_header_id|>\n{instruction}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n{input_text}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n{output_text}<|eot_id|>"
        )
    }


def main():
    login(token=HF_TOKEN)

    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else '없음'}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB" if torch.cuda.is_available() else "")

    print("\n데이터 로드 중...")
    train_ds, val_ds = load_data()
    train_ds = train_ds.map(format_prompt)
    val_ds   = val_ds.map(format_prompt)
    print(f"train: {len(train_ds):,}개  |  val: {len(val_ds):,}개")
    print(f"\n[샘플]\n{train_ds[0]['text'][:400]}\n")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print("모델 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = SFTConfig(
        output_dir=SAVE_PATH,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        fp16=False,
        bf16=True,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        load_best_model_at_end=True,
        report_to="none",
        resume_from_checkpoint=True,
        max_seq_length=MAX_LENGTH,
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    print("\n학습 시작...")
    trainer.train()
    trainer.save_model(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)
    print(f"\n저장 완료: {SAVE_PATH}")


if __name__ == "__main__":
    main()
