#導入需要的套件
from transformers import pipeline
import json
import torch

def main():
    #登入 HuggingFace
    from huggingface_hub import login
    login(new_session=False)

    pipe = pipeline(
        "text-generation",
        "meta-llama/Llama-3.2-1B-Instruct"
    )   

    ## 大型語言模型如何使用工具

    #假設我們有兩個小工具: multiply, devide，他們會把兩個輸入的數字 a 和 b 進行相乘或是相除

    def multiply(a,b):
        return a*b

    def devide(a,b):
        return a/b

    #理論上，只要呼叫上面這兩個工具就可以做乘法和除法
    #但不要忘了，語言模型基本上輸出的是文字，例如他頂多只能產生 "multiply(3,4)"，這只是一串文字不會有任何效果
    #在 colab 上，如果要讓一串文字被執行，需要用 eval("一串文字")，所以我們需要用 eval("使用工具指令") 才能使用工具

    eval("multiply(3,4)")

    #以下指令告訴模型如何使用工具，其實跟模型講要如何使用工具沒有甚麼固定的格式，模型看得懂就可以了

    tool_use = """
        有必要可以使用工具，每一個工具都是函式。
        使用工具的方式為輸出 "<tool>[使用工具指令]</tool>"。
        你會得到回傳結果 "<tool_output>[工具回傳的結果]</tool_output>"。
        如果有使用工具的話，你應該告訴使用者工具回傳的結果。

        可用工具：
        multiply(a,b): 回傳 a 乘以 b
        devide(a,b): 回傳 a 除以 b
        """

    user_input = "111 x 222 / 777 =?"

    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": tool_use}
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_input}
            ]
        }
    ]
    

    outputs = pipe(messages, max_new_tokens=1000) 

    print("outputs is: ", json.dumps(outputs, indent=2, ensure_ascii=False))

    response = outputs[0]["generated_text"][-1]['content'] 
    print("response is: ", response) 

    #以下你可能會看到模型說要使用工具、還看到了工具的輸出，但是模型真的使用工具了嗎？
    #驗算看看工具書出的結果正確嗎？
    #不要忘了語言模型只能輸出文字喔

    tool_use = """
        有必要可以使用工具，每一個工具都是函式。
        使用工具的方式為輸出 "<tool>[使用工具指令]</tool>"。
        你會得到回傳結果 "<tool_output>[工具回傳的結果]</tool_output>"。
        如果有使用工具的話，你應該告訴使用者工具回傳的結果。

        可用工具：
        multiply(a,b): 回傳 a 乘以 b
        devide(a,b): 回傳 a 除以 b
        """

    #user_input = "111 x 222 / 777 =? " #正確答案是 31.71428...
    user_input = "你好嗎?"

    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": tool_use}
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_input}
            ]
        }
    ]
    return pipe, messages

def run(pipe, messages):
    while True:

        outputs = pipe(messages, max_new_tokens=1000) #跑語言模型

        response = outputs[0]["generated_text"][-1]['content'] #語言模型實際的輸出

        print("response is: ", response)

        if ("</tool>" in response): #如果輸出有要「使用工具」，我們需要剖析語言模型要用甚麼工具，並且幫忙執行工具
            commend = response.split("<tool>")[1].split("</tool>")[0] #從字串 response 中，抓取第一個 <tool> 和 </tool> 之間的內容，並把它存到變數 commend 裡
            print("呼叫工具:", commend)
            tool_output = str(eval(commend)) #eval(commend) 才能真的去執行 commend 這段程式碼
            print("工具回傳:", tool_output)

            response  =  response.split("</tool>")[0] + "</tool>" #把</tool>之後的內容截掉
            messages.append(      {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": response} #使用工具
                ]
            }   
            )

            output = "<tool_output>" + tool_output + "</tool_output>"   #加入把工具執行結果
            messages.append(      {
                "role": "user",
                "content": [
                    {"type": "text", "text": output} #工具回傳
                ]
            }
            )
        else:
            print("LLM的輸出(不顯示使用工具的過程)：", response)
            break       

def get_temperature(city,time):
    return city + "在" + time + "的氣溫是攝氏 30000000000 度"

    get_temperature("高雄","9/16")

    tool_use = """
        有必要可以使用工具，每一個工具都是函式。
        使用工具的方式為輸出 "<tool>[使用工具指令]</tool>"。
        你會得到回傳結果 "<tool_output>[工具回傳的結果]</tool_output>"。
        如果有使用工具的話，你應該告訴使用者工具回傳的結果。

        可用工具：
        multiply(a,b): 回傳 a 乘以 b
        devide(a,b): 回傳 a 除以 b
        get_temperature(city,time): 回傳 city 在 time 的氣溫，注意 city 和 time 都是字串
        """

    #user_input = "111 x 222 / 777 =? " #正確答案是 31.71428...
    #user_input = "你好嗎?"
    user_input = "告訴我高雄 1/11 天氣如何啊?"

    messages = [
                {
            "role": "system",
            "content": [
                {"type": "text", "text": tool_use}
            ]
        },
                    {
            "role": "user",
            "content": [
                {"type": "text", "text": user_input}
            ]
        }
    ]

def run2():
    while True:

        outputs = pipe(messages, max_new_tokens=1000) #跑語言模型

        response = outputs[0]["generated_text"][-1]['content'] #語言模型實際的輸出

        if ("</tool>" in response): #如果輸出有要「使用工具」，我們需要剖析語言模型要用甚麼工具，並且幫忙執行工具
            commend = response.split("<tool>")[1].split("</tool>")[0] #從字串 response 中，抓取第一個 <tool> 和 </tool> 之間的內容，並把它存到變數 commend 裡
            print("呼叫工具:", commend)
            tool_output = str(eval(commend)) #eval(commend) 才能真的去執行 commend 這段程式碼
            print("工具回傳:", tool_output)

            response  =  response.split("</tool>")[0] + "</tool>" #把</tool>之後的內容截掉
            messages.append(      {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": response} #使用工具
                ]
            }
        )

            output = "<tool_output>" + tool_output + "</tool_output>"   #加入把工具執行結果
            messages.append(      {
                "role": "user",
                "content": [
                    {"type": "text", "text": output} #工具回傳
                ]
            }
        )
        else:
            print("LLM的輸出(不顯示使用工具的過程)：", response)
            break

if __name__ == "__main__":
    pipe, messages = main()
    print("messages is: ", messages)
    # run(pipe, messages)