from promptLib import createCatalog, countTokens, writehistory, genRANstring
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