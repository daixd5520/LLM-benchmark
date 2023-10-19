import os
import platform
import signal
from transformers import AutoTokenizer, AutoModel
import readline
import time

tokenizer = AutoTokenizer.from_pretrained("/home/ubuntu/workspace/Model-ChatGLM2-6B/chatglm2-6b", trust_remote_code=True)
start_time = time.time()
model = AutoModel.from_pretrained("/home/ubuntu/workspace/Model-ChatGLM2-6B/chatglm2-6b", trust_remote_code=True).cuda()
end_time = time.time()
print(f"Model loading time: {end_time - start_time} seconds")
# 多显卡支持，使用下面两行代替上面一行，将num_gpus改为你实际的显卡数量
# from utils import load_model_on_gpus
# model = load_model_on_gpus("THUDM/chatglm2-6b", num_gpus=2)
model = model.eval()

os_name = platform.system()
clear_command = 'cls' if os_name == 'Windows' else 'clear'
stop_stream = False


def build_prompt(history):
    prompt = "欢迎使用 ChatGLM2-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序"
    for query, response in history:
        prompt += f"\n\n用户：{query}"
        prompt += f"\n\nChatGLM2-6B：{response}"
    return prompt


def signal_handler(signal, frame):
    global stop_stream
    stop_stream = True


def main():
    past_key_values, history = None, []
    global stop_stream
    print("欢迎使用 ChatGLM2-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
    while True:
        query = input("\n用户：")
        if query.strip() == "stop":
            break
        if query.strip() == "clear":
            past_key_values, history = None, []
            os.system(clear_command)
            print("欢迎使用 ChatGLM2-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
            continue
        #######################改：增start#########################
        model.transformer.fwd_num = 0
        model.transformer.seq_len = 0
        model.transformer.encode_time = 0
        model.transformer.decode_time = 0
        total_response = ""
        #######################改：增end###########################
        print("\nChatGLM：", end="")
        current_length = 0

        for response, history, past_key_values in model.stream_chat(tokenizer, query, history=history,
                                                                    past_key_values=past_key_values,
                                                                    return_past_key_values=True):
            if stop_stream:
                stop_stream = False
                break
            else:
                print(response[current_length:], end="", flush=True)
                #print("\n")
                current_length = len(response)
        #######################改：增start#########################
                total_response += response
                #print("                 seq len: %d " % (model.transformer.seq_len))
                #print("                 tokens: %d " % (model.transformer.fwd_num - 1))
                #print("                 encode dur: %.4f ms" % (model.transformer.encode_time))
                #print("                 decode dur: %.4f ms" % (model.transformer.decode_time))
                #print("                 seq len: %d, tokens: %d,encode dur: %.4f ms,decode dur: %.4f ms" % (model.transformer.seq_len,model.transformer.fwd_num - 1,model.transformer.encode_time,model.transformer.decode_time))
        print("\n")
        print("Length of input ids(输入长度): %d" % (model.transformer.seq_len))
        print("Tokens generated in 1 sec（第一秒生成token数）: %d" % (model.transformer.one_sec_tokens))
        print("First token time（生成第一个token耗时）: %.4f ms" % (model.transformer.first_token_time))
        print("Generated token count（总生成token数）:%d" % (model.transformer.fwd_num - 1))
        print("Time per token（平均生成每个token用时）:%.4f ms" % ((model.transformer.encode_time+model.transformer.decode_time)/(model.transformer.fwd_num - 1)))
        model.transformer.one_sec_tokens=0
        #######################改：增end###########################
        print("")


if __name__ == "__main__":
    main()
