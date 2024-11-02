# Chat with an intelligent assistant in your terminal  
# MODEL: https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF
# smollm2-1.7b-instruct-q4_k_m.gguf
# FOR PROMPT TUNING PURPOSES ONLY
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
modelname = 'smollm2-1.7b-instruct-q4_k_m.gguf'
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
logfile = f'smollm2-1.7b-instruct_{genRANstring(5)}_CHAT_log.txt'
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

###############################MODEL CARD###############################################
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
