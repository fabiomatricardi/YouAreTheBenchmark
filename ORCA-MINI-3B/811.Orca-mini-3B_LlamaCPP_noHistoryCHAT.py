# Chat with an intelligent assistant in your terminal  
# REPO https://huggingface.co/Aryanne/Orca-Mini-3B-gguf
# Original model: https://huggingface.co/pankajmathur/orca_mini_3b
# models/q4_1-orca-mini-3b.gguf
#####MODEL WITHOUT CHAT TEMPLATE###############################################################
import sys
from time import sleep
import warnings
warnings.filterwarnings(action='ignore')
import datetime
from promptLib import countTokens, writehistory, createCatalog
from promptLib import genRANstring, createStats
import argparse

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
####################INITIALIZE THE MODEL###################################
stops = ['###']
tasks = createCatalog()
modelname = 'q4_1-orca-mini-3b.gguf'
rootname = 'orca-mini-3B'
# create THE LOG FILE 
coded5 = genRANstring(5)
logfile = f'logs/{rootname}_CHAT_CPP_{coded5}_log.txt'
logfilename = logfile
#Write in the history the first 2 sessions
writehistory(logfilename,f'{str(datetime.datetime.now())}\n\nYour own LocalGPT with 💻 {modelname}\n---\n🧠🫡: You are a helpful assistant.')    
writehistory(logfilename,f'💻: How can I assist you today in writing?')

print("\033[95;3;6m")
print("1. Waiting 10 seconds for the API to load...")
from llama_cpp import Llama
llm = Llama(
            model_path=f'models/{modelname}',
            n_gpu_layers=ngpu_layers,
            temperature=0.1,
            n_ctx=8192,
            max_tokens=1000,
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
system = 'You are a helpful AI assistant.'
instruction = 'Explain the plot of Cinderella in a sentence.'
prompt = f"### System:\n{system}\n\n### User:\n{instruction}\n\n### Response:\n"
print('Question:', instruction)
start = datetime.datetime.now()
print("💻 > ", end="", flush=True)
full_response = ""
fisrtround = 0
for chunk in llm.create_completion(
    prompt,
    temperature=0.25,
    repeat_penalty= 1.31,
    stop=stops,
    max_tokens=1000,
    stream=True,):
    try:
        if chunk['choices'][0]['text']:
            if fisrtround==0:
                print(chunk['choices'][0]['text'], end="", flush=True)
                full_response += chunk['choices'][0]['text']
                ttftoken = datetime.datetime.now() - start  
                fisrtround = 1
            else:
                print(chunk['choices'][0]['text'], end="", flush=True)
                full_response += chunk['choices'][0]['text']                              
    except:
        pass         
delta = datetime.datetime.now() - start
output = full_response
print('')
print("\033[91;1m")
rating = 'PUT IT LATER'#input('Rate from 0 (BAD) to 5 (VERY GOOD) the quality of generation> ')
print("\033[92;1m")
stats = createStats(delta,instruction,output,rating,logfilename,'Alignment Generation')
print(stats)
writehistory(logfilename,f'''👨‍💻 . {instruction}
💻 > {output}
{stats}
''')

############################# START TURN BASED CHAT #################################
print('Starting now Normal Chat turn based interface...')
counter = 1
while True:
    # Reset history every turn
    system = 'You are a helpful AI assistant.'
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
    prompt = userinput
    print("\033[92;1m")
    # Preparing Generation history pair
    # Starting generation loop
    full_response = ""
    fisrtround = 0
    start = datetime.datetime.now()
    print("💻 > ", end="", flush=True)
    full_response = ""
    fisrtround = 0
    for chunk in llm.create_completion(
        prompt,
        temperature=0.25,
        repeat_penalty= 1.31,
        stop=stops,
        max_tokens=1000,
        stream=True,):
        try:
            if chunk['choices'][0]['text']:
                if fisrtround==0:
                    print(chunk['choices'][0]['text'], end="", flush=True)
                    full_response += chunk['choices'][0]['text']
                    ttftoken = datetime.datetime.now() - start  
                    fisrtround = 1
                else:
                    print(chunk['choices'][0]['text'], end="", flush=True)
                    full_response += chunk['choices'][0]['text']                              
        except:
            pass      
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
    stats = createStats(delta,prompt,full_response,rating,logfilename,'CHAT')
    print(stats)
    print(f'>>>⏱️ Time to First Token: {ttofseconds} seconds')
    writehistory(logfilename,f'''👨‍💻 > {userinput}
💻 > {full_response}
{stats}
>>> Time to First Token: {ttofseconds} seconds
''')
    history = []
