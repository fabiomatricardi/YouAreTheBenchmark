# Chat with an intelligent assistant in your terminal  
# https://huggingface.co/Felladrin/gguf-stablelm-2-1_6b-chat/tree/main

# stablelm-2-1_6b-chat.Q8_0.gguf
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
stops = ['<|endoftext|>']
tasks = createCatalog()
modelname = 'stablelm-2-1_6b-chat.Q8_0.gguf'
# create THE LOG FILE 
coded5 = genRANstring(5)
logfile = f'logs/stablelm-2-1_6b-chat_CPP_{coded5}_log.txt'
csvfile = f'logs/stablelm-2-1_6b-chat_CPP_{coded5}.csv'
logfilename = logfile
NCTX = 4096
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
print('\nCSV String:\n', df_csv_data)     



##############MODEL CARD##########################################
"""
https://huggingface.co/Felladrin/gguf-stablelm-2-1_6b-chat/tree/main
stablelm-2-1_6b-chat.Q8_0.gguf

Origianl Model: https://huggingface.co/stabilityai/stablelm-2-1_6b-chat

-----------------------------------------------------------------------

>>> from llama_cpp import Llama
>>> llm = Llama(model_path='models/stablelm-2-1_6b-chat.Q8_0.gguf')
llama_model_loader: loaded meta data with 21 key-value pairs and 340 tensors from models/stablelm-2-1_6b-chat.Q8_0.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
     general.architecture str         = stablelm
             general.name str         = stablelm-2-1_6b-chat
  stablelm.context_length u32         = 4096
stablelm.embedding_length u32         = 2048
llm_load_print_meta: n_ctx_orig_yarn  = 4096
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = stablelm
llm_load_print_meta: vocab type       = BPE
llm_load_print_meta: n_vocab          = 100352
llm_load_print_meta: n_merges         = 100000
llm_load_print_meta: vocab_only       = 0
llm_load_print_meta: n_ctx_train      = 4096
llm_load_print_meta: n_embd           = 2048
llm_load_print_meta: n_layer          = 24
llm_load_print_meta: model type       = 1B
llm_load_print_meta: model ftype      = Q8_0
llm_load_print_meta: model params     = 1.64 B
llm_load_print_meta: model size       = 1.63 GiB (8.50 BPW)
llm_load_print_meta: general.name     = stablelm-2-1_6b-chat
llm_load_print_meta: BOS token        = 100257 '<|endoftext|>'
llm_load_print_meta: EOS token        = 100257 '<|endoftext|>'
llm_load_print_meta: UNK token        = 100257 '<|endoftext|>'
llm_load_print_meta: PAD token        = 100257 '<|endoftext|>'
llm_load_print_meta: LF token         = 128 'Ä'
llm_load_print_meta: EOT token        = 100278 '<|im_end|>'

CHAT TEMPLATE = YES
System message = YES
NCTX = 4096


"""