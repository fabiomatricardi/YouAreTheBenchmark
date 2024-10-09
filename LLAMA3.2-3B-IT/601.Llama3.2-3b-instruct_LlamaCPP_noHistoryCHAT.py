# Chat with an intelligent assistant in your terminal  
# MODEL: https://huggingface.co/Bartowski
# models/Llama-3.2-3B-Instruct-Q5_K_M.gguf
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
modelname = 'Llama-3.2-3B-Instruct-Q5_K_M.gguf'
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
logfile = f'Llama-3.2-3B-Instruct_{genRANstring(5)}_CHAT_log.txt'
logfilename = logfile
#Write in the history the first 2 sessions
writehistory(logfilename,f'{str(datetime.datetime.now())}\n\nYour own LocalGPT with 💻 {modelname}\n---\n🧠🫡: You are a helpful assistant.')    
writehistory(logfilename,f'💻: How can I assist you today in writing?')


STOPS = ['<|eot_id|>']
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

###############################MODEL CARD###############################################
"""
MODEL CARD FOR Llama-3.2-3B-Instruct-Q5_K_M
-------------------------------------------
repo: bartowski_Llama-3.2-3B-Instruct-GGUF · Hugging Face
SIZE: 3.21 B
NCTX = 128K
CHAT TEMPLATE = YES with Cutting Knowledge Date: December 2023 and Today Date
SYSTEM MESSAGE support = YES
FUNCTION CALLING: YES 
LAYERS = 28
STOPS = ['<|eot_id|>']

llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = llama
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = Llama 3.2 3B Instruct
llama_model_loader: - kv   3:                           general.finetune str              = Instruct
llama_model_loader: - kv   4:                           general.basename str              = Llama-3.2
llama_model_loader: - kv   5:                         general.size_label str              = 3B
llama_model_loader: - kv   6:                            general.license str              = llama3.2
llama_model_loader: - kv   7:                               general.tags arr[str,6]       = ["facebook", "meta", "pytorch", "llam...
llama_model_loader: - kv   8:                          general.languages arr[str,8]       = ["en", "de", "fr", "it", "pt", "hi", …

llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = llama
llm_load_print_meta: vocab type       = BPE
llm_load_print_meta: n_vocab          = 128256
llm_load_print_meta: n_merges         = 280147
llm_load_print_meta: vocab_only       = 0
llm_load_print_meta: n_ctx_train      = 131072
llm_load_print_meta: n_embd           = 3072
llm_load_print_meta: n_layer          = 28
llm_load_print_meta: model ftype      = Q5_K - Medium
llm_load_print_meta: model params     = 3.21 B
llm_load_print_meta: model size       = 2.16 GiB (5.76 BPW)
llm_load_print_meta: general.name     = Llama 3.2 3B Instruct
llm_load_print_meta: BOS token        = 128000 '<|begin_of_text|>'
llm_load_print_meta: EOS token        = 128009 '<|eot_id|>'
llm_load_print_meta: LF token         = 128 'Ä'
llm_load_print_meta: EOT token        = 128009 '<|eot_id|>'

Available chat formats from metadata: chat_template.default
Using gguf chat template: {{- bos_token }}
{%- if custom_tools is defined %}
    {%- set tools = custom_tools %}
{%- endif %}
{%- if not tools_in_user_message is defined %}
    {%- set tools_in_user_message = true %}
{%- endif %}
{%- if not date_string is defined %}
    {%- if strftime_now is defined %}
        {%- set date_string = strftime_now("%d %b %Y") %}
    {%- else %}
        {%- set date_string = "26 Jul 2024" %}
    {%- endif %}
{%- endif %}
{%- if not tools is defined %}
    {%- set tools = none %}
{%- endif %}

{#- This block extracts the system message, so we can slot it into the right place. #}
{%- if messages[0]['role'] == 'system' %}
    {%- set system_message = messages[0]['content']|trim %}
    {%- set messages = messages[1:] %}
{%- else %}
    {%- set system_message = "" %}
{%- endif %}

{#- System message #}
{{- "<|start_header_id|>system<|end_header_id|>\n\n" }}
{%- if tools is not none %}
    {{- "Environment: ipython\n" }}
{%- endif %}
{{- "Cutting Knowledge Date: December 2023\n" }}
{{- "Today Date: " + date_string + "\n\n" }}
{%- if tools is not none and not tools_in_user_message %}
    {{- "You have access to the following functions. To call a function, please respond with JSON for a function call." }}
    {{- 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.' }}
    {{- "Do not use variables.\n\n" }}
    {%- for t in tools %}
        {{- t | tojson(indent=4) }}
        {{- "\n\n" }}
    {%- endfor %}
{%- endif %}
{{- system_message }}
{{- "<|eot_id|>" }}

{#- Custom tools are passed in a user message with some extra guidance #}
{%- if tools_in_user_message and not tools is none %}
    {#- Extract the first user message so we can plug it in here #}
    {%- if messages | length != 0 %}
        {%- set first_user_message = messages[0]['content']|trim %}
        {%- set messages = messages[1:] %}
    {%- else %}
        {{- raise_exception("Cannot put tools in the first user message when there's no first user message!") }}
{%- endif %}
    {{- '<|start_header_id|>user<|end_header_id|>\n\n' -}}
    {{- "Given the following functions, please respond with a JSON for a function call " }}
    {{- "with its proper arguments that best answers the given prompt.\n\n" }}
    {{- 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.' }}
    {{- "Do not use variables.\n\n" }}
    {%- for t in tools %}
        {{- t | tojson(indent=4) }}
        {{- "\n\n" }}
    {%- endfor %}
    {{- first_user_message + "<|eot_id|>"}}
{%- endif %}

{%- for message in messages %}
    {%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}
        {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' }}
    {%- elif 'tool_calls' in message %}
        {%- if not message.tool_calls|length == 1 %}
            {{- raise_exception("This model only supports single tool-calls at once!") }}
        {%- endif %}
        {%- set tool_call = message.tool_calls[0].function %}
        {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' -}}
        {{- '{"name": "' + tool_call.name + '", ' }}
        {{- '"parameters": ' }}
        {{- tool_call.arguments | tojson }}
        {{- "}" }}
        {{- "<|eot_id|>" }}
    {%- elif message.role == "tool" or message.role == "ipython" %}
        {{- "<|start_header_id|>ipython<|end_header_id|>\n\n" }}
        {%- if message.content is mapping or message.content is iterable %}
            {{- message.content | tojson }}
        {%- else %}
            {{- message.content }}
        {%- endif %}
        {{- "<|eot_id|>" }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
{%- endif %}

Using chat eos_token: <|eot_id|>
Using chat bos_token: <|begin_of_text|>

"""
