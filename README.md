# YouAreTheBenchmark
Personal Catalog of prompt templates for NLP tasks


## An automated prompt evaluation with YOU as human feedback
- Python file `00.LlamaCPP_autotest.py`
- run a GGUF model with llama-cpp-python
- go thrugh all the NLP tasks

## A python file with prompt catalog and supporting functions
> credit to [stackOverflow](https://stackoverflow.com/questions/139180/how-to-list-all-functions-in-a-module) to print the Classes and Methods
```
import promptLib
help(promptLib)

Help on module promptLib:

NAME
    promptLib

FUNCTIONS
    countTokens(text)
        Use tiktoken to count the number of tokens
        text -> str input
        Return -> int number of tokens counted

    createCatalog()
        Create a dictionary with
        'task'   : description of the NLP task in the prompt
        'prompt' : the instruction prompt for the LLM

    createStats(delta, question, output, rating, logfilename, task)
        Takes in all the generation main info and return KPIs
        delta -> datetime.now() delta
        question -> str the user input to the LLM
        output -> str the generation from the LLM
        rating -> str human eval feedback rating
        logfilename -> str filepath/filename
        task -> str description of the NLP task describing the prompt

    genRANstring(n)
        n = int number of char to randomize

    writehistory(filename, text)
        save a string into a logfile with python file operations
        filename -> str pathfile/filename
        text -> str, the text to be written in the file

FILE
    c:\users\fabiomatricardi\documents\dev\models_autotest\promptlib.py
```

## Usage
Example
```
from promptLib import createCatalog
from promptLib import countTokens
from promptLib import writehistory
from promptLib import genRANstring
from time import sleep

print('1) Create a prompt catalog...')
sleep(4)
tasks = createCatalog()
for items in tasks:
    print("\033[0m")  #reset all
    print("\033[91;1m")
    print(f'TASK>> {items["task"]}')
    print("\033[92;1m")
    print(f'PROMPT>> {items["prompt"]}')
    print("\033[95;3;6m")
    print("---")
print("\033[0m")  #reset all    

print('2) Creating 7 digit random HASH...')
sleep(4)
filename = f'Random-{genRANstring(7)}_TXT.txt'
print(filename)

print('3) Writing hello world in a TXT file...')
sleep(4)
writehistory(filename,'Helloword\n\n')
print('Done')
message = '4) Count the tokens in this message...'
print('4) Count the tokens in this message...')
sleep(4)
print(f'TOKEN COUNT: {countTokens(message)}')
```
