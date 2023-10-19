import os
import torch
import platform
import subprocess
from colorama import Fore, Style
from tempfile import NamedTemporaryFile
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
import time

def init_model():
    print("init model ...")
    model = AutoModelForCausalLM.from_pretrained(
        "/home/ubuntu/workspace/Baichuan2-7B-Chat",
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.generation_config = GenerationConfig.from_pretrained(
        "/home/ubuntu/workspace/Baichuan2-7B-Chat"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "/home/ubuntu/workspace/Baichuan2-7B-Chat",
        use_fast=False,
        trust_remote_code=True
    )
    return model, tokenizer


def clear_screen():
    if platform.system() == "Windows":
        os.system("cls")
    else:
        os.system("clear")
    print(Fore.YELLOW + Style.BRIGHT + "欢迎使用百川大模型，输入进行对话，vim 多行输入，clear 清空历史，CTRL+C 中断生成，stream 开关流式生成，exit 结束。")
    return []


def vim_input():
    with NamedTemporaryFile() as tempfile:
        tempfile.close()
        subprocess.call(['vim', '+star', tempfile.name])
        text = open(tempfile.name).read()
    return text


def main(stream=True):
    model_start=time.time()
    model, tokenizer = init_model()
    model_end=time.time()
    
    messages = clear_screen()
    print("模型加载时间：",model_end-model_start)
    while True:
        prompt = input(Fore.GREEN + Style.BRIGHT + "\n用户：" + Style.NORMAL)
        if prompt.strip() == "exit":
            break
        if prompt.strip() == "clear":
            messages = clear_screen()
            continue
        if prompt.strip() == 'vim':
            prompt = vim_input()
            print(prompt)
        print(Fore.CYAN + Style.BRIGHT + "\nBaichuan 2：" + Style.NORMAL, end='')
        if prompt.strip() == "stream":
            stream = not stream
            print(Fore.YELLOW + "({}流式生成)\n".format("开启" if stream else "关闭"), end='')
            continue
        messages.append({"role": "user", "content": prompt})
        if stream:
            position = 0
            try:
                for response in model.chat(tokenizer, messages, stream=True):
                    print(response[position:], end='', flush=True)
                    
                    position = len(response)
                    if torch.backends.mps.is_available():
                        torch.mps.empty_cache()
            except KeyboardInterrupt:
                pass
            print()
        else:
            response = model.chat(tokenizer, messages)
            print(response)
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        #################改：增start##################
        print("\ninput_length:" ,model.input_len)
        print("\nTokens generated in 1 sec:",model.model.one_sec_tokens)
        print("\nFirst token time(秒):",model.model.first_token_time)
        print("\ngenerated tokens count:",model.model.fwd_num-1)
        #print("\ntime_tot(秒):" ,model.model.time_tot)
        print("\nms/token:",model.model.time_tot*1000/(model.model.fwd_num-1))
        model.model.time_tot=0
        model.model.fwd_num=0
        model.model.one_sec_tokens=0
        #################改：增end####################
        messages.append({"role": "assistant", "content": response})
    print(Style.RESET_ALL)


if __name__ == "__main__":
    main()
