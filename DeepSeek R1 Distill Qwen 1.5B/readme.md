# testing DeepSeek R1 Distill Qwen 1.5B quantized with my RBYF
Using my prompt catalog to verify the performances of the DeepSeek R1 model, the smallest one.

### technology stack
Run the model with llama.cpp binaries, Vulkan flavour

- Download the ZIP archive llama-b4539-bin-win-vulkan-x64.zip from [the official Repo](https://github.com/ggerganov/llama.cpp/releases/download/b4539/llama-b4539-bin-win-vulkan-x64.zip)
- extract the ZIP archive directly in the project directory
- download the GGUF from the [Bartowski repo](https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF) I used the [DeepSeek-R1-Distill-Qwen-1.5B-Q6_K.gguf](https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-1.5B-Q6_K.gguf?download=true)
- open the terminal and run `start cmd.exe /k "llama-server.exe -m DeepSeek-R1-Distill-Qwen-1.5B-Q6_K.gguf -c 15000 -ngl 999"`

As for the python environemnt you need:
- my prommpt library `promptLibv2Qwen.py` from this folder
- `pip install openai`
- the python file `333.DeepSeek R1 Distill Qwen 1.5B_LlamaCPP_API_promptTest.py` from this folder 

From the terminal with the `venv` activated run
```
python 333.DeepSeek R1 Distill Qwen 1.5B_LlamaCPP_API_promptTest.py
```

At the end of every turn you have to assign an evaluation from 0 to 5 with a comment: something like

`3 - we get 3 sentences instead of 2, but the user request was somehow fulfilled`
