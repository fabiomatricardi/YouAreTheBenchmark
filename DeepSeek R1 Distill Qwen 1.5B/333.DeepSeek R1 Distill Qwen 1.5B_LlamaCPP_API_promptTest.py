# Chat with an intelligent assistant in your terminal  
# MODEL: DeepSeek R1 Distill Qwen 1.5B
# Original model: from BARTOWSKI REPO
# DeepSeek-R1-Distill-Qwen-1.5B-Q6_K.gguf
import sys
from time import sleep
import warnings
warnings.filterwarnings(action='ignore')
import datetime
from promptLibv2Qwen import countTokens, writehistory, createCatalog
from promptLibv2Qwen import genRANstring, createStats
import argparse
from openai import OpenAI
## PREPARING FINAL DATASET

pd_id = []
pd_task = []
pd_vote = []
pd_remarks = []
####################Add GPU argument in the parser###################################
# LODADEd AS ASERVER WITH LLAMACPP-SERVER 
####################INITIALIZE THE MODEL###################################
STOPS = ['<´¢£endÔûüofÔûüsentence´¢£>']
tasks = createCatalog()
NCTX = 15000
modelname = 'DeepSeek-R1-Distill-Qwen-1.5B-Q6_K.gguf'
# create THE LOG FILE 
coded5 = genRANstring(5)
logfile = f'logs/DeepSeek-R1-Qwen1.5b_CPP_{coded5}_log.txt'
csvfile = f'logs/DeepSeek-R1-Qwen1.5b_CPP_{coded5}.csv'
logfilename = logfile
#Write in the history the first 2 sessions
writehistory(logfilename,f'''{str(datetime.datetime.now())}
---
Your own LocalGPT with 💻 {modelname}
---
🧠🫡: You are a helpful assistant.
temperature: 0.15
repeat penalty: 1.31
max tokens: 1500
---''')    
writehistory(logfilename,f'💻: How can I assist you today in writing?')
# LOAD THE MODEL
print("\033[95;3;6m")
print("1. Waiting 10 seconds for the API to load...")
llm = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed", organization=modelname)
print(f"2. Model {modelname} loaded with LlamaCPP-server binaries...")
print("\033[0m")  #reset all
history = []
print("\033[92;1m")
print(f'📝Logfile: {logfilename}')
##################### ALIGNMENT FIRST GENERATION ##############################################
question = 'Explain the plot of Cinderella in one sentence.'
test = [
    {"role": "user", "content": question}
]
print('Question:', question)
start = datetime.datetime.now()
print("💻 > ", end="", flush=True)
full_response = ""
fisrtround = 0
completion = llm.chat.completions.create(
    model="local-model", # this field is currently unused
    messages=test,
    temperature=0.25,
    frequency_penalty  = 1.53,
    max_tokens = 1200,
    stream=True,
    stop=STOPS
)
for chunk in completion:
    if chunk.choices[0].delta.content:
        if fisrtround==0:   
            print(chunk.choices[0].delta.content, end="", flush=True)
            full_response += chunk.choices[0].delta.content
            ttftoken = datetime.datetime.now() - start  
            fisrtround = 1
        else:
            print(chunk.choices[0].delta.content, end="", flush=True)
            full_response += chunk.choices[0].delta.content                      
delta = datetime.datetime.now() - start
output = full_response
print('')
print("\033[91;1m")
rating = input('Rate from 0 (BAD) to 5 (VERY GOOD) the quality of generation> ')
print("\033[92;1m")
stats = createStats(delta,question,output,rating,logfilename,'Alignment Generation',ttftoken)
print(stats)
writehistory(logfilename,f'''👨‍💻 . {question}
💻 > {output}
{stats}
''')

############################# AUTOMATIC PROMPTING EVALUATION  11 TURNS #################################
id =1
for items in tasks:
    fisrtround = 0
    task = items["task"]
    prompt = items["prompt"]
    test = []
    print(f'NLP TAKS>>> {task}')
    print("\033[91;1m")  #red
    print(prompt)
    test.append({"role": "user", "content": prompt})
    print("\033[92;1m")
    full_response = ""
    start = datetime.datetime.now()
    print("💻 > ", end="", flush=True)
    completion = llm.chat.completions.create(
        model="local-model", # this field is currently unused
        messages=test,
        temperature=0.25,
        frequency_penalty  = 1.53,
        max_tokens = 1200,
        stream=True,
        stop=STOPS
    )
    for chunk in completion:
        if chunk.choices[0].delta.content:
            if fisrtround==0:   
                print(chunk.choices[0].delta.content, end="", flush=True)
                full_response += chunk.choices[0].delta.content
                ttftoken = datetime.datetime.now() - start  
                fisrtround = 1
            else:
                print(chunk.choices[0].delta.content, end="", flush=True)
                full_response += chunk.choices[0].delta.content          
    delta = datetime.datetime.now() - start
    print('')
    print("\033[91;1m")
    rating = input('Rate from 0 (BAD) to 5 (VERY GOOD) the quality of generation> ')
    print("\033[92;1m")
    stats = createStats(delta,prompt,full_response,rating,logfilename,task,ttftoken)
    print(stats)
    writehistory(logfilename,f'''👨‍💻 > {prompt}
💻 > {full_response}
{stats}
''')
    pd_id.append(id)
    pd_task.append(task)
    pd_vote.append(rating[:2])
    pd_remarks.append(rating[2:])
    id += 1
# create dataframe and save to csv
zipped = list(zip(pd_id,pd_task,pd_vote,pd_remarks))
import pandas as pdd
df = pdd.DataFrame(zipped, columns=['#', 'TASK', 'VOTE','REMARKS'])
#saving the DataFrame as a CSV file 
df_csv_data = df.to_csv(csvfile, index = False, encoding='utf-8') 
print('\nCSV String:\n', df)     
from rich.console import Console
console = Console()
console.print('---')
console.print(df)   



##############MODEL CARD##########################################
"""
# Chat with an intelligent assistant in your terminal  
# MODEL: DeepSeek R1 Distill Qwen 1.5B
# Original model: from BARTOWSKI REPO
# DeepSeek-R1-Distill-Qwen-1.5B-Q6_K.gguf
MODELCARD
===========================================================
mp = 'DeepSeek-R1-Distill-Qwen-1.5B-Q6_K.gguf'


CHAT TEMPLATE = yes
NCTX = 131072

Prompt format
```
ggml_vulkan: Found 1 Vulkan devices:
ggml_vulkan: 0 = NVIDIA GeForce RTX 2060 with Max-Q Design (NVIDIA) | uma: 0 | fp16: 1 | warp size: 32 | matrix cores: KHR_coopmat
build: 4539 (564804b7) with MSVC 19.42.34436.0 for x64
system info: n_threads = 8, n_threads_batch = 8, total_threads = 16

system_info: n_threads = 8 (n_threads_batch = 8) / 16 | CPU : SSE3 = 1 | SSSE3 = 1 | AVX = 1 | AVX2 = 1 | F16C = 1 | FMA = 1 | LLAMAFILE = 1 | OPENMP = 1 | AARCH64_REPACK = 1 |

main: HTTP server is listening, hostname: 127.0.0.1, port: 8080, http threads: 15
main: loading model
srv    load_model: loading model 'D:\LLM-Small\DeepSeek-R1-Distill-Qwen-1.5B-Q6_K.gguf'
llama_model_load_from_file_impl: using device Vulkan0 (NVIDIA GeForce RTX 2060 with Max-Q Design) - 5955 MiB free
llama_model_loader: loaded meta data with 30 key-value pairs and 339 tensors from D:\LLM-Small\DeepSeek-R1-Distill-Qwen-1.5B-Q6_K.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = qwen2
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = DeepSeek R1 Distill Qwen 1.5B
llama_model_loader: - kv   3:                           general.basename str              = DeepSeek-R1-Distill-Qwen
llama_model_loader: - kv   4:                         general.size_label str              = 1.5B
llama_model_loader: - kv   5:                          qwen2.block_count u32              = 28
llama_model_loader: - kv   6:                       qwen2.context_length u32              = 131072
llama_model_loader: - kv   7:                     qwen2.embedding_length u32              = 1536
llama_model_loader: - kv   8:                  qwen2.feed_forward_length u32              = 8960
llama_model_loader: - kv   9:                 qwen2.attention.head_count u32              = 12
llama_model_loader: - kv  10:              qwen2.attention.head_count_kv u32              = 2
llama_model_loader: - kv  11:                       qwen2.rope.freq_base f32              = 10000.000000
llama_model_loader: - kv  12:     qwen2.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  13:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  14:                         tokenizer.ggml.pre str              = deepseek-r1-qwen
llama_model_loader: - kv  15:                      tokenizer.ggml.tokens arr[str,151936]  = ["!", "\"", "#", "$", "%", "&", "'", ...
llama_model_loader: - kv  16:                  tokenizer.ggml.token_type arr[i32,151936]  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  17:                      tokenizer.ggml.merges arr[str,151387]  = ["─á ─á", "─á─á ─á─á", "i n", "─á t",...
llama_model_loader: - kv  18:                tokenizer.ggml.bos_token_id u32              = 151646
llama_model_loader: - kv  19:                tokenizer.ggml.eos_token_id u32              = 151643
llama_model_loader: - kv  20:            tokenizer.ggml.padding_token_id u32              = 151643
llama_model_loader: - kv  21:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  22:               tokenizer.ggml.add_eos_token bool             = false
llama_model_loader: - kv  23:                    tokenizer.chat_template str              = {% if not add_generation_prompt is de...
llama_model_loader: - kv  24:               general.quantization_version u32              = 2
llama_model_loader: - kv  25:                          general.file_type u32              = 18
llama_model_loader: - kv  26:                      quantize.imatrix.file str              = /models_out/DeepSeek-R1-Distill-Qwen-...
llama_model_loader: - kv  27:                   quantize.imatrix.dataset str              = /training_dir/calibration_datav3.txt
llama_model_loader: - kv  28:             quantize.imatrix.entries_count i32              = 196
llama_model_loader: - kv  29:              quantize.imatrix.chunks_count i32              = 128
llama_model_loader: - type  f32:  141 tensors
llama_model_loader: - type q6_K:  198 tensors
print_info: file format = GGUF V3 (latest)
print_info: file type   = Q6_K
print_info: file size   = 1.36 GiB (6.56 BPW)
load: special_eos_id is not in special_eog_ids - the tokenizer config may be incorrect
load: special tokens cache size = 22
load: token to piece cache size = 0.9310 MB
print_info: arch             = qwen2
print_info: vocab_only       = 0
print_info: n_ctx_train      = 131072
print_info: n_embd           = 1536
print_info: n_layer          = 28
print_info: n_head           = 12
print_info: n_head_kv        = 2
print_info: n_rot            = 128
print_info: n_swa            = 0
print_info: n_embd_head_k    = 128
print_info: n_embd_head_v    = 128
print_info: n_gqa            = 6
print_info: n_embd_k_gqa     = 256
print_info: n_embd_v_gqa     = 256
print_info: f_norm_eps       = 0.0e+00
print_info: f_norm_rms_eps   = 1.0e-06
print_info: f_clamp_kqv      = 0.0e+00
print_info: f_max_alibi_bias = 0.0e+00
print_info: f_logit_scale    = 0.0e+00
print_info: n_ff             = 8960
print_info: n_expert         = 0
print_info: n_expert_used    = 0
print_info: causal attn      = 1
print_info: pooling type     = 0
print_info: rope type        = 2
print_info: rope scaling     = linear
print_info: freq_base_train  = 10000.0
print_info: freq_scale_train = 1
print_info: n_ctx_orig_yarn  = 131072
print_info: rope_finetuned   = unknown
print_info: ssm_d_conv       = 0
print_info: ssm_d_inner      = 0
print_info: ssm_d_state      = 0
print_info: ssm_dt_rank      = 0
print_info: ssm_dt_b_c_rms   = 0
print_info: model type       = 1.5B
print_info: model params     = 1.78 B
print_info: general.name     = DeepSeek R1 Distill Qwen 1.5B
print_info: vocab type       = BPE
print_info: n_vocab          = 151936
print_info: n_merges         = 151387
print_info: BOS token        = 151646 '<´¢£beginÔûüofÔûüsentence´¢£>'
print_info: EOS token        = 151643 '<´¢£endÔûüofÔûüsentence´¢£>'
print_info: EOT token        = 151643 '<´¢£endÔûüofÔûüsentence´¢£>'
print_info: PAD token        = 151643 '<´¢£endÔûüofÔûüsentence´¢£>'
print_info: LF token         = 148848 '├ä─¼'
print_info: FIM PRE token    = 151659 '<|fim_prefix|>'
print_info: FIM SUF token    = 151661 '<|fim_suffix|>'
print_info: FIM MID token    = 151660 '<|fim_middle|>'
print_info: FIM PAD token    = 151662 '<|fim_pad|>'
print_info: FIM REP token    = 151663 '<|repo_name|>'
print_info: FIM SEP token    = 151664 '<|file_sep|>'
print_info: EOG token        = 151643 '<´¢£endÔûüofÔûüsentence´¢£>'
print_info: EOG token        = 151662 '<|fim_pad|>'
print_info: EOG token        = 151663 '<|repo_name|>'
print_info: EOG token        = 151664 '<|file_sep|>'
print_info: max token length = 256
ggml_vulkan: Compiling shaders....................................................Done!
load_tensors: offloading 28 repeating layers to GPU
load_tensors: offloading output layer to GPU
load_tensors: offloaded 29/29 layers to GPU
load_tensors:      Vulkan0 model buffer size =  1208.10 MiB
load_tensors:   CPU_Mapped model buffer size =   182.57 MiB
llama_init_from_model: n_seq_max     = 1
llama_init_from_model: n_ctx         = 8192
llama_init_from_model: n_ctx_per_seq = 8192
llama_init_from_model: n_batch       = 2048
llama_init_from_model: n_ubatch      = 512
llama_init_from_model: flash_attn    = 0
llama_init_from_model: freq_base     = 10000.0
llama_init_from_model: freq_scale    = 1
llama_init_from_model: n_ctx_per_seq (8192) < n_ctx_train (131072) -- the full capacity of the model will not be utilized
llama_kv_cache_init: kv_size = 8192, offload = 1, type_k = 'f16', type_v = 'f16', n_layer = 28, can_shift = 1
llama_kv_cache_init:    Vulkan0 KV buffer size =   224.00 MiB
llama_init_from_model: KV self size  =  224.00 MiB, K (f16):  112.00 MiB, V (f16):  112.00 MiB
llama_init_from_model: Vulkan_Host  output buffer size =     0.58 MiB
llama_init_from_model:    Vulkan0 compute buffer size =   299.75 MiB
llama_init_from_model: Vulkan_Host compute buffer size =    19.01 MiB
llama_init_from_model: graph nodes  = 986
llama_init_from_model: graph splits = 2
common_init_from_params: setting dry_penalty_last_n to ctx_size = 8192
common_init_from_params: warming up the model with an empty run - please wait ... (--no-warmup to disable)
srv          init: initializing slots, n_slots = 1
slot         init: id  0 | task -1 | new slot n_ctx_slot = 8192
main: model loaded
main: chat template, chat_template: {% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set ns = namespace(is_first=false, is_tool=false, is_output_first=true, system_prompt='') %}{%- for message in messages %}{%- if message['role'] == 'system' %}{% set ns.system_prompt = message['content'] %}{%- endif %}{%- endfor %}{{bos_token}}{{ns.system_prompt}}{%- for message in messages %}{%- if message['role'] == 'user' %}{%- set ns.is_tool = false -%}{{'<´¢£User´¢£>' + message['content']}}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is none %}{%- set ns.is_tool = false -%}{%- for tool in message['tool_calls']%}{%- if not ns.is_first %}{{'<´¢£Assistant´¢£><´¢£toolÔûücallsÔûübegin´¢£><´¢£toolÔûücallÔûübegin´¢£>' + tool['type'] + '<´¢£toolÔûüsep´¢£>' + tool['function']['name'] + '\n' + '```json' + '\n' + tool['function']['arguments'] + '\n' + '```' + '<´¢£toolÔûücallÔûüend´¢£>'}}{%- set ns.is_first = true -%}{%- else %}{{'\n' + '<´¢£toolÔûücallÔûübegin´¢£>' + tool['type'] + '<´¢£toolÔûüsep´¢£>' + tool['function']['name'] + '\n' + '```json' + '\n' + tool['function']['arguments'] + '\n' + '```' + '<´¢£toolÔûücallÔûüend´¢£>'}}{{'<´¢£toolÔûücallsÔûüend´¢£><´¢£endÔûüofÔûüsentence´¢£>'}}{%- endif %}{%- endfor %}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is not none %}{%- if ns.is_tool %}{{'<´¢£toolÔûüoutputsÔûüend´¢£>' + message['content'] + '<´¢£endÔûüofÔûüsentence´¢£>'}}{%- set ns.is_tool = false -%}{%- else %}{% set content = message['content'] %}{% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}{{'<´¢£Assistant´¢£>' + content + '<´¢£endÔûüofÔûüsentence´¢£>'}}{%- endif %}{%- endif %}{%- if message['role'] == 'tool' %}{%- set ns.is_tool = true -%}{%- if ns.is_output_first %}{{'<´¢£toolÔûüoutputsÔûübegin´¢£><´¢£toolÔûüoutputÔûübegin´¢£>' + message['content'] + '<´¢£toolÔûüoutputÔûüend´¢£>'}}{%- set ns.is_output_first = false %}{%- else %}{{'\n<´¢£toolÔûüoutputÔûübegin´¢£>' + message['content'] + '<´¢£toolÔûüoutputÔûüend´¢£>'}}{%- endif %}{%- endif %}{%- endfor -%}{% if ns.is_tool %}{{'<´¢£toolÔûüoutputsÔûüend´¢£>'}}{% endif %}{% if add_generation_prompt and not ns.is_tool %}{{'<´¢£Assistant´¢£>'}}{% endif %}, example_format: 'You are a helpful assistant

<´¢£User´¢£>Hello<´¢£Assistant´¢£>Hi there<´¢£endÔûüofÔûüsentence´¢£><´¢£User´¢£>How are you?<´¢£Assistant´¢£>'
main: server is listening on http://127.0.0.1:8080 - starting the main loop
srv  update_slots: all slots are idle

"""