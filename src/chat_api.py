from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from langchain_core.prompts import ChatPromptTemplate
from bs4 import BeautifulSoup
import torch
from peft import PeftModel

def print_outputs(outputs):
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    print("-" * 80)


with open ("../data/md_files/123.md") as f:
	data=f.read()


#peft_model_id = "rycont/kanana-2.1b-lora-reasoning"
#soup = BeautifulSoup(data,"html.parser")
#text = soup.get_text()
text=data


lora_model_path = "./lora_adapter_ka"
base_model_name = "kakaocorp/kanana-nano-2.1b-instruct"
#base_model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
#base_model_name = "binwon/kanana-cot
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
#model = PeftModel.from_pretrained(base_model, lora_model_path)

if tokenizer.pad_token is None:
	tokenizer.pad_token = tokenizer.eos_token


streamer = TextStreamer(tokenizer, skip_prompt=True,)

SYSTEM_PROMPT = """
You are a university assistant chatbot with expert knowledge about academic programs, administration, campus life, and student services.
Your role is to provide concise, accurate, and student-friendly answers using the retrieved context. response to korean
"""


c = """ [Source: https://software.cbnu.ac.kr/sub0401/2103]\n– 11:00 / 14:00 - 18:00 4. 장 소 : 소프트웨어학과 사무실 (S1-4 219호)\n\n[Source: http://software.cbnu.ac.kr/sub0505]\n산학프로젝트\n\n개요\n\n찾아오시는길\n\n개인정보처리방침\n\n이메일무단수집거부\n\n사이트맵\n\n충북대학교 소프트웨어학부\n\n인공지능 전공ㆍ소프트웨어 전공 (28644) 충북 청주시 서원구 충대로 1, 충북대학교 전자정보대학 소프트웨어학부 S4-1동(전자정보 3관) 217호 Copyright © SOFTWARE. All Rights Reserved.\n\n[Source: https://software.cbnu.ac.kr/sub0505]\n산학프로젝트\n\n개요\n\n찾아오시는길\n\n개인정보처리방침\n\n이메일무단수집거부\n\n사이트맵\n\n충북대학교 소프트웨어학부\n\n인공지능 전공ㆍ소프트웨어 전공 (28644) 충북 청주시 서원구 충대로 1, 충북대학교 전자정보대학 소프트웨어학부 S4-1동(전자정보 3관) 217호 Copyright © SOFTWARE. All Rights Reserved.\n\n[Source: https://software.cbnu.ac.kr/sub0401/2486]\n제목+내용제목내용댓글이름닉네임아이디태그 학부 2018.07.02 14:23 소프트웨어학과 학과사무실 이전 안내 =============================================================== 관리자 조회 수 470 추천 수 0 댓글 0 ? 크게 작게 ? 크게 작게 소프트웨어학과 학과 사무실을 S4-1동 217호로 이전하였습니다. 2018학년도 제2학기 생활관비 납부 및 결원 충원 추가모집 안내 2018.07.03by\n\n[Source: http://software.cbnu.ac.kr/sub0103]\n학부소개\n\n연혁\n\nSOFTWARE HISTORY\n\n2011년\n\n소프트웨어학과 설립\n\n2013년\n\n서울어코드사업 선정(2012.07~2019.02)\n\n2014년\n\n융합학과군 디지털정보융합학과 통합\n\n2015년\n\n지역선도대학 육성사업 수행(2014.10~2019.02)\n\n충북대학교 우수학과, 우수교육프로그램 선정\n\n2017년\n\n충북대학교 우수학과, 우수교육프로그램 선정\n\n2019년\n\nSW 중심대학 사업 선정, 충북대학교 최우수학과, 우수교육프로그램 선정
"""
q="소프트웨어학과 위치"
user_prompt=f"""Below is a question with retrieved context. 
Use the given context to answer.

### Context ###
{c}

### Question ###
{q}

"""
#user_prompt = "### Question ###\n{q}"
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": user_prompt}
]
#input_ids = tokenizer.apply_chat_template(
#    messages,
#    tokenize=True,
#    continue_final_message=True,
#    return_tensors="pt"
#).to("cuda")

#input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
input_ids = tokenizer.apply_chat_template(messages, continue_final_message=True ,return_tensors="pt").to("cuda")
#_ = model.eval()

with torch.no_grad():
	outputs = model.generate(
					input_ids, 
					max_new_tokens=1024,
					top_p=0.9,
					#num_beams=5,
					early_stopping=True,
					streamer = streamer,
					tokenizer = tokenizer,
					eos_token_id=tokenizer.eos_token_id,
					pad_token_id=tokenizer.pad_token_id,
					#stop_strings = ["</answer>",]
			)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
