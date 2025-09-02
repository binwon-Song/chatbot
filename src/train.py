import wandb
from unsloth import is_bfloat16_supported
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments
import os

train_data= "../data/qa/qa_clean.json"
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
model_name = "kakaocorp/kanana-nano-2.1b-instruct"
os.environ["WANDB_PROJECT"] = "cbnu-chatbot"
os.environ["WANDB_ENTITY"] = "binwon-song"


#train_config={
#	"learning_rate": 2e-4,
#	"epoches" : 10,
#	"batch_size" : 32,
#}

base_model = AutoModelForCausalLM.from_pretrained(
	model_name,
	torch_dtype=torch.bfloat16,
	device_map="auto",
	trust_remote_code=True
)

args = SFTConfig(
	per_device_train_batch_size=64,
	gradient_accumulation_steps=4,
	warmup_steps = 4,
	num_train_epochs = 2, # Set this for 1 full training run.
	learning_rate = 2e-4,
	fp16 = not is_bfloat16_supported(),
	bf16 = is_bfloat16_supported(),
	optim = "adamw_8bit",
	weight_decay = 0.01,
	lr_scheduler_type = "linear",
	seed = 3407,
	report_to = "wandb",
	max_length = 2048,
	dataset_num_proc = 4,
	packing = True, # Can make training 5x faster for short sequences.
	)

lora_config = LoraConfig(
	task_type = "CAUSAL_LM",
	r=8,
	lora_alpha = 16,
	lora_dropout = 0.1,
	target_modules = [
		"q_proj",
		"k_proj",
		"v_proj",
		"o_proj",
		"gate_proj",
		"up_proj",
		"down_proj",],
)

run = wandb.init(
		entity="binwon-song",
		project = "cbnu-chatbot",
		job_type = "training",
	)


tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side = "right")
EOS_TOKEN = tokenizer.eos_token
tokenizer.pad_token = tokenizer.eos_token

dataset = Dataset.from_json(train_data)


def tokenize_function(examples):
	tokens = tokenizer(examples["text"], padding=True, return_tensors="pt")
	tokens["labels"] = tokens["input_ids"]
	return tokens

def formatting_prompts_func(examples):
	train_prompt_style="""
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Below is an instruction that describes a task, paired with an input that provides further context. 
Write a response that appropriately completes the request. 
Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
context: {}
{}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
{}
source: {}
<|end_of_text|>
"""
	inputs = examples["question"]
	context = examples["search_result"]
	outputs = examples["llm_result"]
	source = examples["extracted_ref_numbers"]
	texts = []
	for i,c,o,s in zip(inputs,context, outputs, source):
		text = train_prompt_style.format(c, i , o , s) + EOS_TOKEN
		texts.append(text)
	return {
		"text": texts,
	}
	


def formatting_prompts_deep(examples):
	prompt_style = """Below is an instruction that describes a task, paired with a question and retrieved context from university-related resources.
Use the given context to generate a helpful and accurate response for the student. 
Before answering, think step by step inside <think> tags, then provide the final answer clearly.

### Instruction:
You are a university assistant chatbot with expert knowledge about academic programs, administration, campus life, and student services. 
Your role is to provide concise, accurate, and student-friendly answers using the retrieved context. 
If the context is insufficient,You should say 'I don't know'.

### Question:
{question}

### Retrieved Context:
{context}

### Response:
<think>
{cot}
</think>

Answer:
{answer}
"""
	texts = []
	for q, c, cot, a in zip(examples["question"], examples["context"], examples["CoT"], examples["answer"]):
		text = prompt_style.format(
					question=q,
					context=c,
					cot=cot,
					answer=a
			) + EOS_TOKEN

		texts.append(text)
	return {
		"text": texts,
	}



def formatting_prompts_func_old(examples):
	alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
	### Question:
	{}

	### Answer:
	{}
"""
	instructions = examples["Question"]
	outputs = examples["CoT_Rationale"]

	EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN

	texts = []
	for instruction, output in zip(instructions, outputs):
		# Must add EOS_TOKEN, otherwise your generation will go on forever!
		text = alpaca_prompt.format(instruction, output) + EOS_TOKEN
		texts.append(text)

	return {"text": texts}


dataset = dataset.map(formatting_prompts_deep, batched=True)
#train_val = dataset.train_test_split(test_size=0, seed=3407)
#
#train_dataset= train_val["train"]
#eval_dataset= train_val["test"]
train_dataset = dataset
train_dataset = train_dataset.map(tokenize_function, batched=True)
#eval_dataset = eval_dataset.map(tokenize_function, batched=True)

#tokenized_dataset = dataset.map(tokenize_function, batched=True)

model = get_peft_model(base_model, lora_config)

trainer = SFTTrainer(
	model=model,
	processing_class = tokenizer,
	train_dataset = train_dataset,
	#eval_dataset = eval_dataset,
	args = args
)

print('Count of using GPUs:', torch.cuda.device_count())   #출력결과: 2 (2, 3 두개 사용하므로)
print('Current cuda device:', torch.cuda.current_device())  # 출력결과: 2 (2, 3 중 앞의 GPU #2 의미)

trainer_stats = trainer.train()
model.save_pretrained("./lora_adapter_ka")
tokenizer.save_pretrained("./lora_adapter_ka")

