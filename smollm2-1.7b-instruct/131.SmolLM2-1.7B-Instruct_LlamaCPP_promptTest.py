# Chat with an intelligent assistant in your terminal  
# MODEL: https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF
# smollm2-1.7b-instruct-q4_k_m.gguf
import sys
from time import sleep
import warnings
warnings.filterwarnings(action='ignore')
import datetime
from promptLibv2Qwen import countTokens, writehistory, createCatalog
from promptLibv2Qwen import genRANstring, createStats
import argparse
## PREPARING FINAL DATASET

pd_id = []
pd_task = []
pd_vote = []
pd_remarks = []
####################Add GPU argument in the parser###################################
parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gpu", type=int, default=0,nargs='?',
                    help="The number of layers to load on GPU")
args = parser.parse_args()
if args.gpu == None:
   ngpu_layers = 0 
else:
    ngpu_layers = args.gpu
print(f'Selected GPU: offloading {ngpu_layers} layers...')   
####################INITIALIZE THE MODEL###################################
stops = ['<|im_end|>']
tasks = createCatalog()
NCTX = 8192
modelname = 'smollm2-1.7b-instruct-q4_k_m.gguf'
# create THE LOG FILE 
coded5 = genRANstring(5)
logfile = f'logs/SmolLM2-1.7B-Instruct_CPP_{coded5}_log.txt'
csvfile = f'logs/SmolLM2-1.7B-Instruct_CPP_{coded5}.csv'
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
from llama_cpp import Llama
llm = Llama(
            model_path=f'models/{modelname}',
            n_gpu_layers=ngpu_layers,
            temperature=0.1,
            n_ctx=NCTX,
            max_tokens=1500,
            repeat_penalty=1.178,
            stop=stops,
            verbose=False,
            )
print(f"2. Model {modelname} loaded with LlamaCPP...")
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
for chunk in llm.create_chat_completion(
    messages=test,
    temperature=0.25,
    repeat_penalty= 1.31,
    stop=stops,
    max_tokens=800,
    stream=True,):
    try:
        if chunk["choices"][0]["delta"]["content"]:
            if fisrtround==0:
                print(chunk["choices"][0]["delta"]["content"], end="", flush=True)
                full_response += chunk["choices"][0]["delta"]["content"]
                ttftoken = datetime.datetime.now() - start  
                fisrtround = 1
            else:
                print(chunk["choices"][0]["delta"]["content"], end="", flush=True)
                full_response += chunk["choices"][0]["delta"]["content"]                              
    except:
        pass        
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
    for chunk in llm.create_chat_completion(
        messages=test,
        temperature=0.15,
        repeat_penalty= 1.31,
        stop=stops,
        max_tokens=1000,
        stream=True,):
        try:
            if chunk["choices"][0]["delta"]["content"]:
                if fisrtround==0:
                    print(chunk["choices"][0]["delta"]["content"], end="", flush=True)
                    full_response += chunk["choices"][0]["delta"]["content"]
                    ttftoken = datetime.datetime.now() - start  
                    fisrtround = 1
                else:
                    print(chunk["choices"][0]["delta"]["content"], end="", flush=True)
                    full_response += chunk["choices"][0]["delta"]["content"]                              
        except:
            pass        
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
# MODEL: https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF
# smollm2-1.7b-instruct-q4_k_m.gguf
MODELCARD
===========================================================
NAMe: Smollm2 1.7B 8k Mix7 Ep2 v2
CONTEXT WINDOW: 8192
CHAT TEMPLATE = Yes
PARAMTERES = 1.7 B
LAYERS = 24
stops= '<|im_end|>'
SYSTEMPROMPT  Available


>>> mp = 'models/smollm2-1.7b-instruct-q4_k_m.gguf'
>>> from llama_cpp import Llama
>>> llm = Llama(model_path=mp)
llama_model_loader: loaded meta data with 34 key-value pairs and 218 tensors from models/smollm2-1.7b-instruct-q4_k_m.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
            general.architecture str              = llama
                    general.type str              = model
                    general.name str              = Smollm2 1.7B 8k Mix7 Ep2 v2
                 general.version str              = v2
            general.organization str              = Loubnabnl
                general.finetune str              = 8k-mix7-ep2
                general.basename str              = smollm2
              general.size_label str              = 1.7B
                 general.license str              = apache-2.0
               general.languages arr[str,1]       = ["en"]
               llama.block_count u32              = 24
            llama.context_length u32              = 8192
          llama.embedding_length u32              = 2048
       llama.feed_forward_length u32              = 8192
      llama.attention.head_count u32              = 32
   llama.attention.head_count_kv u32              = 32
            llama.rope.freq_base f32              = 130000.000000
attention.layer_norm_rms_epsilon f32              = 0.000010
               general.file_type u32              = 15
                llama.vocab_size u32              = 49152
      llama.rope.dimension_count u32              = 64
 tokenizer.ggml.add_space_prefix bool             = false
    tokenizer.ggml.add_bos_token bool             = false
            tokenizer.ggml.model str              = gpt2
              tokenizer.ggml.pre str              = smollm
           tokenizer.ggml.tokens arr[str,49152]   = ["<|endoftext|>", "<|im_start|>", "<|...
       tokenizer.ggml.token_type arr[i32,49152]   = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, ...
           tokenizer.ggml.merges arr[str,48900]   = ["Ġ t", "Ġ a", "i n", "h e", "Ġ Ġ...
     tokenizer.ggml.bos_token_id u32              = 1
     tokenizer.ggml.eos_token_id u32              = 2
 tokenizer.ggml.unknown_token_id u32              = 0
 tokenizer.ggml.padding_token_id u32              = 2
         tokenizer.chat_template str              = {% for message in messages %}{% if lo...
    general.quantization_version u32              = 2
llama_model_loader: - type  f32:   49 tensors
llama_model_loader: - type q4_K:  144 tensors
llama_model_loader: - type q6_K:   25 tensors
llm_load_vocab: special tokens cache size = 17
llm_load_vocab: token to piece cache size = 0.3170 MB
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = llama
llm_load_print_meta: vocab type       = BPE
llm_load_print_meta: n_vocab          = 49152
llm_load_print_meta: n_merges         = 48900
llm_load_print_meta: vocab_only       = 0
llm_load_print_meta: n_ctx_train      = 8192
llm_load_print_meta: n_embd           = 2048
llm_load_print_meta: n_layer          = 24
llm_load_print_meta: n_head           = 32
llm_load_print_meta: n_head_kv        = 32
llm_load_print_meta: n_rot            = 64
llm_load_print_meta: n_swa            = 0
llm_load_print_meta: n_embd_head_k    = 64
llm_load_print_meta: n_embd_head_v    = 64
llm_load_print_meta: n_gqa            = 1
llm_load_print_meta: n_embd_k_gqa     = 2048
llm_load_print_meta: n_embd_v_gqa     = 2048
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-05
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: f_logit_scale    = 0.0e+00
llm_load_print_meta: n_ff             = 8192
llm_load_print_meta: n_expert         = 0
llm_load_print_meta: n_expert_used    = 0
llm_load_print_meta: causal attn      = 1
llm_load_print_meta: pooling type     = 0
llm_load_print_meta: rope type        = 0
llm_load_print_meta: rope scaling     = linear
llm_load_print_meta: freq_base_train  = 130000.0
llm_load_print_meta: freq_scale_train = 1
llm_load_print_meta: n_ctx_orig_yarn  = 8192
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: ssm_d_conv       = 0
llm_load_print_meta: ssm_d_inner      = 0
llm_load_print_meta: ssm_d_state      = 0
llm_load_print_meta: ssm_dt_rank      = 0
llm_load_print_meta: ssm_dt_b_c_rms   = 0
llm_load_print_meta: model type       = ?B
llm_load_print_meta: model ftype      = Q4_K - Medium
llm_load_print_meta: model params     = 1.71 B
llm_load_print_meta: model size       = 1005.01 MiB (4.93 BPW)
llm_load_print_meta: general.name     = Smollm2 1.7B 8k Mix7 Ep2 v2
llm_load_print_meta: BOS token        = 1 '<|im_start|>'
llm_load_print_meta: EOS token        = 2 '<|im_end|>'
llm_load_print_meta: UNK token        = 0 '<|endoftext|>'
llm_load_print_meta: PAD token        = 2 '<|im_end|>'
llm_load_print_meta: LF token         = 143 'Ä'
llm_load_print_meta: EOT token        = 0 '<|endoftext|>'
"""