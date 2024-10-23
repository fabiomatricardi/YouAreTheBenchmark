# Chat with an intelligent assistant in your terminal  
# MODEL: ollama-granite3dense
# this wil run granite3-2B-instruct through ollamaAPI
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
modelname = 'granite3-dense:2b'
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