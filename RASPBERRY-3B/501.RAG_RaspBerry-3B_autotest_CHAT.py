# Chat with an intelligent assistant in your terminal  
# MODEL: https://huggingface.co/Qwen
# models/raspberry-3B.Q5_K_M.gguf
import sys
from time import sleep
import warnings
warnings.filterwarnings(action='ignore')
import datetime
import random
import string
import tiktoken
import argparse

#Add GPU argument in the parser
parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gpu", action="store_true")

args = parser.parse_args()
GPU = args.gpu
if GPU:
    ngpu_layers = 28
    print(f'Selected GPU: offloading {ngpu_layers} layers...')
else:
     ngpu_layers = 0   
     print('Loading Model on CPU only......')



encoding = tiktoken.get_encoding("cl100k_base") #context_count = len(encoding.encode(yourtext))
modelname = 'raspberry-3B.Q5_K_M.gguf'
def countTokens(text):
    encoding = tiktoken.get_encoding("cl100k_base") #context_count = len(encoding.encode(yourtext))
    numoftokens = len(encoding.encode(text))
    return numoftokens

def writehistory(filename,text):
    # save the user/assistant pairs into a logfile located in the logs subdirectory
    with open(f'logs/{filename}', 'a', encoding='utf-8') as f:
        f.write(text)
        f.write('\n')
    f.close()

def genRANstring(n):
    """
    n = int number of char to randomize
    """
    N = n
    res = ''.join(random.choices(string.ascii_uppercase +
                                string.digits, k=N))
    return res

# create THE LOG FILE 
logfile = f'raspberry-3B_{genRANstring(5)}_CHAT_log.txt'
logfilename = logfile
#Write in the history the first 2 sessions
writehistory(logfilename,f'{str(datetime.datetime.now())}\n\nYour own LocalGPT with 💻 {modelname}\n---\n🧠🫡: You are a helpful assistant.')    
writehistory(logfilename,f'💻: How can I assist you today in writing?')


STOPS = ['<|im_end|>']
print("\033[95;3;6m")
print("1. Waiting 10 seconds for the API to load...")
from llama_cpp import Llama
llm = Llama(
            model_path=f'models/{modelname}',
            n_gpu_layers=ngpu_layers,
            temperature=1.1,
            top_p = 0.5,
            n_ctx=8192,
            max_tokens=1500,
            repeat_penalty=1.178,
            stop=STOPS,
            verbose=False,
            )
print(f"2. Model {modelname} loaded with LlamaCPP...")
print("\033[0m")  #reset all
history = []
print("\033[92;1m")
print(f'📝Logfile: {logfilename}')


############################# START TURN BASED CHAT #################################
print('Starting now Normal Chat turn based interface...')
counter = 1
while True:
    # Reset history ALWAys
    history = []        
    userinput = ""
    print("\033[1;30m")  #dark grey
    print("Enter your text (end input with Ctrl+D on Unix or Ctrl+Z on Windows) - type quit! to exit the chatroom:")
    print("\033[91;1m")  #red
    lines = sys.stdin.readlines()
    for line in lines:
        userinput += line + "\n"
    if "quit!" in lines[0].lower():
        print("\033[0mBYE BYE!")
        break
    history.append({"role": "user", "content": userinput})
    print("\033[92;1m")
    # Preparing Generation history pair
    new_message = {"role": "assistant", "content": ""}
    # Starting generation loop
    full_response = ""
    fisrtround = 0
    start = datetime.datetime.now()
    print("💻 > ", end="", flush=True)
    firsttoken = 0
    for chunk in llm.create_chat_completion(
        messages=history,
        temperature=0.15,
        repeat_penalty= 1.131,
        stop=STOPS,
        max_tokens=1500,
        stream=True,):
        try:
            if chunk["choices"][0]["delta"]["content"]:
                if firsttoken == 0:
                    ttft = datetime.datetime.now() - start
                    ttft_seconds = ttft.total_seconds()
                    firsttoken = 1
                    print(chunk["choices"][0]["delta"]["content"], end="", flush=True)
                    full_response += chunk["choices"][0]["delta"]["content"]  
                else:
                    print(chunk["choices"][0]["delta"]["content"], end="", flush=True)
                    full_response += chunk["choices"][0]["delta"]["content"]                                                 
        except:
            pass        
    new_message["content"] = full_response
    history.append(new_message)  
    counter += 1  
    delta = datetime.datetime.now() - start
    totalseconds = delta.total_seconds()
    output = full_response
    prompttokens = countTokens(userinput)
    assistanttokens = countTokens(output)
    totaltokens = prompttokens + assistanttokens
    speed = totaltokens/totalseconds
    genspeed = assistanttokens/totalseconds
    print('')
    print("\033[91;1m")
    rating = input('Rate from 0 (BAD) to 5 (VERY GOOD) the quality of generation> ')
    print("\033[92;1m")
    stats = f'''---
Prompt Tokens: {prompttokens}
Output Tokens: {assistanttokens}
TOTAL Tokens: {totaltokens}
>>>⚡Time to first token: {ttft_seconds:.2f} seconds
>>>⏱️Inference time: {delta}
>>>🧮Inference speed: {speed:.3f}  t/s
>>>🏃‍♂️Generation speed: {genspeed:.3f}  t/s
📝Logfile: {logfilename}
>>>💚User rating: {rating}

'''
    print(stats)
    writehistory(logfilename,f'''👨‍💻: {userinput}
💻 > {full_response}
{stats}
📝Logfile: {logfilename}''')

##############################MODEL CARD###############################################
"""
MODEL CARD FOR raspberry-3B.Q5_K_M.gguf
---------------------------------------
repo: mradermacher_raspberry-3B-GGUF · Hugging Face

MODEL SIZE = 3.09B
BASE MODEL: Qwen2.5-3B
NCTX = 32K
CHAT TEMPLATE =  YES
FUNCTION CALLING = YES
SYSTEM MESSAGE SUPPORT = YES
LAYERS = 36
STOPS = ['<|im_end|>']

general.architecture str              = qwen2
        general.type str              = model
        general.name str              = Qwen2.5 3B
general.organization str              = Qwen
    general.basename str              = Qwen2.5
  general.size_label str              = 3B
     general.license str              = other
general.license.name str              = qwen-research
general.license.link str              = https://huggingface.co/Qwen/Qwen2.5-3...
ral.base_model.count u32              = 1
al.base_model.0.name str              = Qwen2.5 3B
model.0.organization str              = Qwen
ase_model.0.repo_url str              = https://huggingface.co/Qwen/Qwen2.5-3B
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = qwen2
llm_load_print_meta: vocab type       = BPE
llm_load_print_meta: n_vocab          = 151936
llm_load_print_meta: n_merges         = 151387
llm_load_print_meta: vocab_only       = 0
llm_load_print_meta: n_ctx_train      = 32768
llm_load_print_meta: n_embd           = 2048
llm_load_print_meta: n_layer          = 36
llm_load_print_meta: model ftype      = Q5_K - Medium
llm_load_print_meta: model params     = 3.09 B
llm_load_print_meta: model size       = 2.07 GiB (5.75 BPW)
llm_load_print_meta: general.name     = Qwen2.5 3B
llm_load_print_meta: BOS token        = 151644 '<|im_start|>'
llm_load_print_meta: EOS token        = 151645 '<|im_end|>'
llm_load_print_meta: PAD token        = 151643 '<|endoftext|>'
llm_load_print_meta: LF token         = 148848 'ÄĬ'
llm_load_print_meta: EOT token        = 151645 '<|im_end|>'
Available chat formats from metadata: chat_template.default
Using gguf chat template: {%- if tools %}
    {{- '<|im_start|>system\n' }}
    {%- if messages[0]['role'] == 'system' %}
        {{- messages[0]['content'] }}
    {%- else %}
        {{- 'You are a helpful assistant.' }}
    {%- endif %}
    {{- "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" }}
    {%- for tool in tools %}
        {{- "\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n" }}
{%- else %}
    {%- if messages[0]['role'] == 'system' %}
        {{- '<|im_start|>system\n' + messages[0]['content'] + '<|im_end|>\n' }}
    {%- else %}
        {{- '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
{%- for message in messages %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) or (message.role == "assistant" and not message.tool_calls) %}
        {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "assistant" %}
        {{- '<|im_start|>' + message.role }}
        {%- if message.content %}
            {{- '\n' + message.content }}
        {%- endif %}
        {%- for tool_call in message.tool_calls %}
            {%- if tool_call.function is defined %}
                {%- set tool_call = tool_call.function %}
            {%- endif %}
            {{- '\n<tool_call>\n{"name": "' }}
            {{- tool_call.name }}
            {{- '", "arguments": ' }}
            {{- tool_call.arguments | tojson }}
            {{- '}\n</tool_call>' }}
        {%- endfor %}
        {{- '<|im_end|>\n' }}
    {%- elif message.role == "tool" %}
        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\n<tool_response>\n' }}
        {{- message.content }}
        {{- '\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
{%- endif %}

Using chat eos_token: <|im_end|>
Using chat bos_token: <|im_start|>



"""
