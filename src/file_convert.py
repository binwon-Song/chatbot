import os
import glob
import re
from bs4 import BeautifulSoup
import markdown
from markdownify import markdownify as md
from pyhtml2pdf import converter
from xhtml2pdf import pisa
from io import BytesIO
import pdfkit

NOISE_PATTERNS = [
    r"위로", r"아래로",
    r"댓글로 가기", r"인쇄",
    r"첨부", r"^\|.*\|.*\|.*$",
    r"Prev", r"Next", r"검색", r"[s,S]earch",
    r"단축키"
]

def is_noise(line: str) -> bool:
    return any(re.search(p, line) for p in NOISE_PATTERNS)


def convert_html_dir_to_md(input_dir='../data/html_files', output_dir='../data/md_files'):
    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(input_dir):
        if file.endswith('.html'):
            html_path = os.path.join(input_dir, file)
            md_filename = os.path.splitext(file)[0] + '.md'
            md_path = os.path.join(output_dir, md_filename)

            with open(html_path, 'r', encoding='utf-8') as f:
                html_content = f.read()

            markdown_content = md(html_content)

            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)

            print(f"✅ Converted: {html_path} -> {md_path}")

def convert_re_md(idir="../data/md_files",odir="../data/md_re_files"):
    os.makedirs(odir,exist_ok=True)
    
    for file in os.listdir(idir):
        if file.endswith('.md'):
            org_path = os.path.join(idir,file)
            new_name = os.path.splitext(file)[0] + '.md'
            new_path = os.path.join(odir, new_name)
            result = open(new_path,"w")
            with open(org_path,'r',encoding = 'utf-8') as f:
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
            print(f"✅ Converted: {org_path} -> {new_path}")

def convert_html_to_pdf(idir,odir):
    os.makedirs(odir,exist_ok=True)
    #pdfkit.from_file(idr,odir,options={'enable-local-file-access':''})
    #pdfkit.from_file(idr,odir,options={'no-images':''})
 
#    try:
#        file_list = sorted(glob.glob(f'{idir}/*.json'))
#        if len(file_list) == 0:
#            return False
#        for f in file_list:
#            pdfDocument= pdf.Document("{f}",options)
#            pdfDocument.save("html_test.pdf")
#        return True
#    except Exception as err:
#        print("[EXCEPTION]",err)
#        return False
            
if __name__ == "__main__":
    idr="../data/1.html"
    odr="./test.pdf"
    #convert_html_dir_to_md()
    #convert_re_md()
    convert_html_to_pdf(idr,odr)
