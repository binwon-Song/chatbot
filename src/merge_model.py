from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_name = "kakaocorp/kanana-nano-2.1b-base"
lora_model_path = "./lora_adapter_ka"
save_path = "../model/kanana-rag"

base_model = AutoModelForCausalLM.from_pretrained(base_model_name, trust_remote_code=True)

model = PeftModel.from_pretrained(base_model, lora_model_path)

model = model.merge_and_unload()  # 이렇게 하면 LoRA weight가 base model에 합쳐집니다.
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

hub_model_id = "kanana-rag"
model.push_to_hub(hub_model_id, token=True)
tokenizer.push_to_hub(hub_model_id, token=True)
