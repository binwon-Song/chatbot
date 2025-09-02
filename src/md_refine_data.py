from langchain_text_splitters import RecursiveCharacterTextSplitter
import markdown
from bs4 import BeautifulSoup
import re

result = open("./test_md2-1.md","w")
NOISE_PATTERNS = [
	r"위로", r"아래로",
	r"댓글로 가기", r"인쇄",
	r"첨부", r"^\|.*\|.*\|.*$",
	r"Prev", r"Next", r"검색", r"[s,S]earch",
	r"단축키"
]

def is_noise(line: str) -> bool:
    return any(re.search(p, line) for p in NOISE_PATTERNS)

with open ("../data/md_files/221.md","r") as f:
	lines = f.readlines()
	cnt=1
	body_start=False
	for line in lines:
		if "[관리]" in line:
			print("body END")
			body_start=False

		if line:
			if cnt==1:
				result.write(line)
			elif line[0]=="#" and not body_start:
				print("body START")
				body_start=True
			if body_start and not is_noise(line.strip()) and line!='\n':
				result.write(line)

		cnt+=1

result.close()

text_splitter = RecursiveCharacterTextSplitter(
    # 청크 크기를 매우 작게 설정합니다. 예시를 위한 설정입니다.
    chunk_size=250,
    # 청크 간의 중복되는 문자 수를 설정합니다.
    chunk_overlap=50,
    # 문자열 길이를 계산하는 함수를 지정합니다.
    length_function=len,
    # 구분자로 정규식을 사용할지 여부를 설정합니다.
    is_separator_regex=False,
)


with open("./test_md2.md","r") as f:
	a=f.read()
	html = markdown.markdown(a)
	soup = BeautifulSoup(html,"html.parser")
	text=soup.get_text()
	print(text_splitter.split_text(text))

