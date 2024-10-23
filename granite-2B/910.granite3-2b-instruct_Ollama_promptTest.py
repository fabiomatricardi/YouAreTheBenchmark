# Chat with an intelligent assistant in your terminal  
# MODEL: ollama-granite3dense
# this wil run granite3-2B-instruct through ollamaAPI
# sources: https://github.com/fabiomatricardi/-LLM-Studies/raw/main/00.consoleAPI_stream.py
# https://github.com/fabiomatricardi/-LLM-Studies/blob/main/01.st-API-openAI_stream.py
# OLLAMA MODEL CARD: https://ollama.com/library/granite3-dense/blobs/604785e698e9
# OPenAI API for Ollama: https://github.com/ollama/ollama/blob/main/docs/openai.md
# https://github.com/ibm-granite/granite-3.0-language-models
# https://www.ibm.com/granite/docs/
# HUGGINFACE: https://huggingface.co/ibm-granite/granite-3.0-2b-instruct
#####################################################################################################

"""
> ollama show granite3-dense
  Model
    architecture        granite
    parameters          2.6B
    context length      4096
    embedding length    2048
    quantization        Q4_K_M

  License
    Apache License
    Version 2.0, January 2004
"""
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
stops = ['<|end_of_text|>']
tasks = createCatalog()
modelname = 'granite3-dense-2b'
rootname = 'granite3-dense-2b-it'
# create THE LOG FILE 
coded5 = genRANstring(5)
logfile = f'logs/{rootname}_OLLAMA_{coded5}_log.txt'
csvfile = f'logs/{rootname}_OLLAMA_{coded5}.csv'
logfilename = logfile
#Write in the history the first 2 sessions
writehistory(logfilename,f'{str(datetime.datetime.now())}\n\nYour own LocalGPT with 💻 {modelname}\n---\n🧠🫡: You are a helpful assistant.')    
writehistory(logfilename,f'💻: How can I assist you today in writing?')
# LOAD THE MODEL
print("\033[95;3;6m")
print("1. Waiting 10 seconds for the API to load...")
client = OpenAI(base_url='http://localhost:11434/v1/', api_key='ollama')
print(f"2. Model {modelname} loaded with LlamaCPP...")
print("\033[0m")  #reset all
history = []
print("\033[92;1m")
print(f'📝Logfile: {logfilename}')
##################### ALIGNMENT FIRST GENERATION ##############################################
question = 'Explain the plot of Cinderella in a sentence.'
test = [
    {"role": "user", "content": question}
]
print('Question:', question)
start = datetime.datetime.now()
print("💻 > ", end="", flush=True)
full_response = ""
completion = client.chat.completions.create(
    messages=test,
    model='granite3-dense',
    temperature=0.25,
    frequency_penalty  = 1.178,
    stop=stops,
    max_tokens=1500,
    stream=True
)
for chunk in completion:
    try:
        if chunk.choices[0].delta.content:
            if fisrtround==0:
                print(chunk.choices[0].delta.content, end="", flush=True)
                full_response += chunk.choices[0].delta.content   
                ttftoken = datetime.datetime.now() - start  
                fisrtround = 1
            else:
                print(chunk.choices[0].delta.content, end="", flush=True)
                full_response += chunk.choices[0].delta.content                                                              
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
    completion = client.chat.completions.create(
        messages=history,
        model='granite3-dense',
        temperature=0.25,
        frequency_penalty  = 1.178,
        stop=stops,
        max_tokens=1500,
        stream=True
    )
    for chunk in completion:
        try:
            if chunk.choices[0].delta.content:
                if fisrtround==0:
                    print(chunk.choices[0].delta.content, end="", flush=True)
                    full_response += chunk.choices[0].delta.content    
                    ttftoken = datetime.datetime.now() - start  
                    fisrtround = 1
                else:
                    print(chunk.choices[0].delta.content, end="", flush=True)
                    full_response += chunk.choices[0].delta.content                                                                                          
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

###############################MODEL CARD###############################################
"""
Granite-3.0-2B-Instruct
Model Summary: Granite-3.0-2B-Instruct is a 2B parameter model finetuned from Granite-3.0-2B-Base using a combination of open source instruction datasets with permissive license and internally collected synthetic datasets. This model is developed using a diverse set of techniques with a structured chat format, including supervised finetuning, model alignment using reinforcement learning, and model merging.

  Model
    architecture        granite
    parameters          2.6B
    context length      4096
    embedding length    2048
    quantization        Q4_K_M

  License
    Apache License
    Version 2.0, January 2004


Developers: Granite Team, IBM
GitHub Repository: ibm-granite/granite-3.0-language-models
Website: Granite Docs
Paper: Granite 3.0 Language Models
Release Date: October 21st, 2024
License: Apache 2.0
Supported Languages: English, German, Spanish, French, Japanese, Portuguese, Arabic, Czech, Italian, Korean, Dutch, and Chinese. Users may finetune Granite 3.0 models for languages beyond these 12 languages.

Intended use: The model is designed to respond to general instructions and can be used to build AI assistants for multiple domains, including business applications.

Capabilities

Summarization
Text classification
Text extraction
Question-answering
Retrieval Augmented Generation (RAG)
Code related tasks
Function-calling tasks
Multilingual dialog use cases


Introduction to Granite 3.0 Language Models
Granite 3.0 language models are a new set of lightweight state-of-the-art, open foundation models that natively support multilinguality, coding, reasoning, and tool usage, including the potential to be run on constrained compute resources. All the models are publicly released under an Apache 2.0 license for both research and commercial use. The models' data curation and training procedure were designed for enterprise usage and customization in mind, with a process that evaluates datasets for governance, risk and compliance (GRC) criteria, in addition to IBM's standard data clearance process and document quality checks.

Granite 3.0 includes 4 different models of varying sizes:

Dense Models: 2B and 8B parameter models, trained on 12 trillion tokens in total.
Mixture-of-Expert (MoE) Models: Sparse 1B and 3B MoE models, with 400M and 800M activated parameters respectively, trained on 10 trillion tokens in total.
Accordingly, these options provide a range of models with different compute requirements to choose from, with appropriate trade-offs with their performance on downstream tasks. At each scale, we release a base model — checkpoints of models after pretraining, as well as instruct checkpoints — models finetuned for dialogue, instruction-following, helpfulness, and safety.

"""
