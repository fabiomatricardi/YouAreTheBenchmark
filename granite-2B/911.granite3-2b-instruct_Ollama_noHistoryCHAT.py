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
from promptLib import countTokens, writehistory, createCatalog
from promptLib import genRANstring, createStats
import argparse
from openai import OpenAI

#Add GPU argument in the parser
parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gpu", action="store_true")

args = parser.parse_args()
GPU = args.gpu
if GPU:
    ngpu_layers = 2
    print(f'Selected GPU: offloading {ngpu_layers} layers...')
else:
     ngpu_layers = 0   #out of 28
     print('Loading Model on CPU only......')

stops = ['<|end_of_text|>']
tasks = createCatalog()
modelname = 'granite3-dense-2b'
# create THE LOG FILE 
logfile = f'logs/{modelname}_CHAT_OLLAMA_{genRANstring(5)}_log.txt'
logfilename = logfile
#Write in the history the first 2 sessions
writehistory(logfilename,f'{str(datetime.datetime.now())}\n\nYour own LocalGPT with 💻 {modelname}\n---\n🧠🫡: You are a helpful assistant.')    
writehistory(logfilename,f'💻: How can I assist you today in writing?')

print("\033[95;3;6m")
print("1. Waiting 10 seconds for the API to load...")
# using OpenAI library to connect to Ollama API endpoint
client = OpenAI(base_url='http://localhost:11434/v1/', api_key='ollama')
print(f"2. Model {modelname} loaded with OLLAMA...")
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
            print(chunk.choices[0].delta.content, end="", flush=True)
            full_response += chunk.choices[0].delta.content                              
    except:
        pass        
delta = datetime.datetime.now() - start
output = full_response
print('')
print("\033[91;1m")
rating = 'PUT IT LATER'#input('Rate from 0 (BAD) to 5 (VERY GOOD) the quality of generation> ')
print("\033[92;1m")
stats = createStats(delta,question,output,rating,logfilename,'Alignment Generation')
print(stats)
writehistory(logfilename,f'''👨‍💻 . {question}
💻 > {output}
{stats}
''')

############################# START TURN BASED CHAT #################################
print('Starting now Normal Chat turn based interface...')
counter = 1
while True:
    # Reset history every turn
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
    new_message["content"] = full_response
    history.append(new_message)  
    counter += 1  
    delta = datetime.datetime.now() - start
    ttofseconds = ttftoken.total_seconds()
    deltaseconds = delta.total_seconds()
    print('')
    print("\033[91;1m")
    print(f'Generation time: {deltaseconds} seconds')
    print(f'Time to First Token: {ttofseconds} seconds')
    rating = input('Rate from 0 (BAD) to 5 (VERY GOOD) the quality of generation> ')
    print("\033[92;1m")
    stats = createStats(delta,userinput,full_response,rating,logfilename,'CHAT')
    print(stats)
    print(f'>>>⏱️ Time to First Token: {ttofseconds} seconds')
    writehistory(logfilename,f'''👨‍💻 > {userinput}
💻 > {full_response}
{stats}
>>> Time to First Token: {ttofseconds} seconds
''')
    history = []

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
