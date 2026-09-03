import os
import base64
from openai import OpenAI
import pandas as pd

os.chdir(os.path.dirname(__file__))

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"), 
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def get_base64_encode(image_url):
    with open(image_url, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

def get_resp_url(image_url_list):
    # 给每一项拼接前缀 ./image/
    prefix = "https://vl-image.oss-cn-shanghai.aliyuncs.com/"
    suffix = ".jpg"
    image_url_list = [prefix + item + suffix for item in image_url_list]
    return image_url_list

def get_resp_base64(image_url_list):
    prefix = "./image/"
    suffix = ".jpg"
    image_url_list = [prefix + item + suffix for item in image_url_list]

    img_list = []
    for image_url in image_url_list:
        base64_encode = get_base64_encode(image_url)

        img_list.append(f"data:image/jpeg;base64,{base64_encode}")
    
    return img_list

# 调用VLM，得到推理结果
# user_prompt：用户想要分析的内容
# image_url：想要分析的图片
def get_response(user_prompt, image_url, is_base64: bool):
    # 得到image_url_list，一张图片也放到[]中
    if image_url.startswith('[') and ',' in image_url:
        # 属于image_url list
        image_url = image_url.strip()
        image_url = image_url[1:-1]
        image_url_list = image_url.split(',')
        image_url_list = [temp_url.strip() for temp_url in image_url_list]
    else:
        image_url_list = [image_url]

    if (is_base64):
        image_url_list = get_resp_base64(image_url_list)
    else:
        image_url_list = get_resp_url(image_url_list)
    # 得到messages
    content = [{"type": "text", "text": f"{user_prompt}"}]
    for temp_url in image_url_list: 
        image_url = temp_url
        content.append({"type": "image_url","image_url": {"url": f"{image_url}"}})
    messages=[{
                "role": "user",
                "content": content
            }
        ]

    print(f'messages={messages}')
    completion = client.chat.completions.create(
        model="qwen-vl-max", #qwen-vl-plus
        messages=messages    
        )
    #print(completion.model_dump_json())
    return completion

df = pd.read_excel('./excel/prompt_template_cn.xlsx')
df['response'] = ''
for index, row in df.iterrows():
    user_prompt = row['prompt']
    image_url = row['image']
    # 得到VLM推理结果
    completion = get_response(user_prompt, image_url, True)
    response = completion.choices[0].message.content
    df.loc[index, 'response'] = response
    print(f"{index+1} {user_prompt} {image_url}")
df.to_excel('./prompt_template_cn_result_base64.xlsx', index=False)
exit()
