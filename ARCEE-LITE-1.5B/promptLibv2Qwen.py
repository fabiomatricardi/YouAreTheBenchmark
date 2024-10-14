"""
V2 changes
added Time To First Token in the statistics ttft
added some more prompts in the catalog
- say 'I am ready'
- modified for Llama3.2-1b Write in a list the three main key points -  format output

20240929 FAMA
"""

import random
import string
import tiktoken

def createCatalog():
    """
    Create a dictionary with 
    'task'   : description of the NLP task in the prompt
    'prompt' : the instruction prompt for the LLM
    """
    context = """One of the things everybody in the West knows about China is that it is not a democracy, and is instead a regime run with an iron fist by a single entity, the Chinese Communist Party, whose leadership rarely acts transparently, running the country without the need for primary elections, alternative candidacies, etc.
In general, those of us who live in democracies, with relatively transparent electoral processes, tend to consider the Chinese system undesirable, little more than a dictatorship where people have no say in who governs them.
That said, among the “advantages” of the Chinese system is that because the leadership never has to put its legitimacy to the vote, it can carry out very long-term planning in the knowledge that another administration isn’t going to come along and change those plans.
Obviously, I put “advantages” in quotation marks because, as democrats, most of my readers would never be willing to sacrifice their freedom for greater planning, but there is no doubt that China, since its system works like this and its population seems to have accepted it for generations, intends to turn this into a comparative advantage, the term used in business when analyzing companies.
It turns out that China’s capacity for long-term planning is achieving something unheard of in the West: it seems the country reached peak carbon dioxide and greenhouse gas emissions in 2023, and that the figures for 2024, driven above all by a determined increase in the installation of renewable energies, are not only lower, but apparently going to mark a turning point.
China and India were until recently the planet’s biggest polluters, but they now offer a model for energy transition (there is still a long way to go; but we are talking about models, not a done deal).
It could soon be the case that the so-called developing countries will be showing the West the way forward."""
    catalog = []
    prmpt_tasks = ["introduction",
               "explain in one sentence",
               "explain in three paragraphs",
               "say 'I am ready'",
               "summarize",
               "Summarize in two sentences",
               "Write in a list the three main key points -  format output",
               "Table of Contents",
               "RAG",
               "Truthful RAG",
               "write content from a reference",
               "extract 5 topics",
               "Creativity: 1000 words SF story",
               "Reflection prompt"
               ]
    prmpt_coll = [
"""Hi there I am Fabio, a Medium writer. who are you?""",
"""explain in one sentence what is science.\n""",
"""explain only in three paragraphs what is artificial intelligence.\n""",
f"""read the following text and when you are done say "I am ready".

[text]
{context}
[end of text]

""",
f"""summarize the following text:
[text]
{context}
[end of text]
write the summary using the information provided.

Summary:
""",
f"""Write a two-sentences summary of the following text:
[text]
{context}
[end of text]
Remember: write only the summary with the text provided.
""",
f"""1. extract the three key points from the provided text
2. format the output as a python list ["point 1","point 2", "point 3"]
[text]
{context}
[end of text]
Return only the python list.
""",
f"""A "table of content" provides an organized structure to the text, allowing readers quick access to key points. Write the "table of content" of the following text. 
[text]
{context}
[end of text]

"table of content":
""",
f"""Reply to the question only using the provided context. If the answer is not contained in the text say "unanswerable".

question: what China achieved with it's long-term planning?

[context]
{context}
[end of context]

answer:
""",
f"""Reply to the question only using the provided context. If the answer is not contained in the provided context say "unanswerable".

question: who is Anne Frank?

[context]
{context}
[end of context]

Remember: if you cannot answer based on the provided context, say "unanswerable"

answer:
""", 

f"""Using the following text as a reference, write a 5-paragraphs essay about "the benefits of China economic model".

[text]
{context}
[end of text]
Remember: use the information provided and write exactly 5 paragraphs.
""",
f"""List five most important topics from the following text:
[text]
{context}
[end of text]
""",
"""Science Fiction: The Last Transmission - Write a story that takes place entirely within a spaceship's cockpit as the sole surviving crew member attempts to send a final message back to Earth before the ship's power runs out. The story should explore themes of isolation, sacrifice, and the importance of human connection in the face of adversity. 800-1000 words.

""",
"""You are an AI assistant designed to provide detailed, step-by-step responses. Your outputs should follow this structure:
1. Begin with a <thinking> section.
2. Inside the thinking section:
   a. Briefly analyze the question and outline your approach.
   b. Present a clear plan of steps to solve the problem.
   c. Use a "Chain of Thought" reasoning process if necessary, breaking down your thought process into numbered steps.
3. Include a <reflection> section for each idea where you:
   a. Review your reasoning.
   b. Check for potential errors or oversights.
   c. Confirm or adjust your conclusion if necessary.
4. Be sure to close all reflection sections.
5. Close the thinking section with </thinking>.
6. Provide your final answer in an <output> section.
Always use these tags in your responses. Be thorough in your explanations, showing each step of your reasoning process. Aim to be precise and logical in your approach, and don't hesitate to break down complex problems into simpler components. Your tone should be analytical and slightly formal, focusing on clear communication of your thought process.
Remember: Both <thinking> and <reflection> MUST be tags and must be closed at their conclusion
Make sure all <tags> are on separate lines with no other text. Do not include other text on a line containing a tag.

user question: explain why it is crucial for teachers to learn how to use generative AI for their job and for the future of education. Include relevant learning path for teachers and educators. 

"""
]
    for i in range(0,len(prmpt_tasks)):
        catalog.append({'task':prmpt_tasks[i],
                        'prompt': prmpt_coll[i]})
    return catalog

def countTokens(text):
    """
    Use tiktoken to count the number of tokens
    text -> str input
    Return -> int number of tokens counted
    """
    encoding = tiktoken.get_encoding("cl100k_base") #context_count = len(encoding.encode(yourtext))
    numoftokens = len(encoding.encode(text))
    return numoftokens

def writehistory(filename,text):
    """
    save a string into a logfile with python file operations
    filename -> str pathfile/filename
    text -> str, the text to be written in the file
    """
    with open(f'{filename}', 'a', encoding='utf-8') as f:
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

def createStats(delta,question,output,rating,logfilename,task,ttft):
    """
    Takes in all the generation main info and return KPIs
    delta -> datetime.now() delta
    question -> str the user input to the LLM
    output -> str the generation from the LLM
    rating -> str human eval feedback rating
    logfilename -> str filepath/filename
    task -> str description of the NLP task describing the prompt
    ttft -> datetime.now() delta time to first token
    """
    totalseconds = delta.total_seconds()
    prompttokens = countTokens(question)
    assistanttokens = countTokens(output)
    totaltokens = prompttokens + assistanttokens
    speed = totaltokens/totalseconds
    genspeed = assistanttokens/totalseconds
    ttofseconds = ttft.total_seconds()
    stats = f'''---
Prompt Tokens: {prompttokens}
Output Tokens: {assistanttokens}
TOTAL Tokens: {totaltokens}
>>>⏱️ Time to First Token: {ttofseconds} seconds
>>>⏱️ Inference time:   {delta}
>>>🧮 Inference speed:  {speed:.3f}  t/s
>>>🏍️ Generation speed: {genspeed:.3f}  t/s
>>>📝 Logfile:     {logfilename}
>>>💚 User rating: {rating}
>>>✅ NLP TAKS:    {task}
'''
    return stats