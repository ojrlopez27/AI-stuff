
# source: https://github.com/MikeyBeez/SimpleAgent/blob/main/ollama/RAGtool3.py

# from langchain_community.llms import Ollama
import ollama
from langchain_community.document_loaders import WebBaseLoader, JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings, GPT4AllEmbeddings, HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
import gradio as gr
import time


# Initialize variables to store the previous URL and its corresponding data and embeddings
prev_filename = None
data = None
vectorstore = None

template = "You are an AI system that performs NLU processing. Given a user utterance (\"input\"), you will generate the " \
           "corresponding user intent (\"user_intent\") and extract the main entity (\"extracted_entity\"). There are 17 " \
           "unique user intents as described in the context. I am providing examples for each user intent and entities. " \
           "So, given an input, you will compare that input against the description of each intent, and will use the examples " \
           "to detemine the most similar one. If no intent is matched, then return only the intent \"unknown\". " \
           "Given the input \"{prompt}\" generate the corresponding output (make sure to generte ONLY the JSON output, " \
           "do not generate additional text):" \

qachain = None
retriever = None


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# Define the Ollama LLM function
def ollama_llm(question, context):
    formatted_prompt = f"Question: {question}\n\nContext: {context}"
    response = ollama.chat(model='mistral:instruct',
                           messages=[{'role': 'user', 'content': formatted_prompt}],
                           options={
                               'num_gpu': 32,
                               'main_gpu': 0,
                               'num_thread': 48,
                               'temperature': 0.0,
                               'seed': 42
                               # 'top_k': 10,
                               # 'top_p': 0.5,
                           })
    return response['message']['content']


def rag_chain(filename: str, question: str):
    global prev_filename, data, vectorstore, qachain, retriever
    if filename != prev_filename:
        # loader = WebBaseLoader(url)
        loader = JSONLoader(file_path=filename, jq_schema=".intents[]", text_content=False)
        data = loader.load()
        prev_filename = filename

        # chunk_size and chunk_overlap are measured in 'characters' (not tokens)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=0)
        docs = text_splitter.split_documents(data)

        # source: https://github.com/langchain-ai/langchain/issues/2026
        # print(f"You have {len(data)} document")
        # print(f"You have {len(data[0].page_content)} characters in that document")
        # for i in docs:
        #     page_con = i.page_content
        #     print('page content length: ', len(page_con))
        #     print('page content: ', page_con)

        embeddings=OllamaEmbeddings(model='llama2',
                                    num_gpu=32,
                                    num_thread=48,
                                    model_kwargs={'device': 'cuda:0'})
                                   # temperature=0.0)
                                   # mirostat_eta=0.5)
        # embeddings = OllamaEmbeddings() #model='mistral:instruct')

        # load the documents into Chroma
        vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)
        retriever = vectorstore.as_retriever()

    # Convert the question into embeddings
    retrieved_docs = retriever.invoke(question)
    formatted_context = format_docs(retrieved_docs)
    response = ollama_llm(question, formatted_context)
    results = {'user_intent': '', 'extracted_entity': ''}
    for key in results.keys():
        pos = response.find('"{}":'.format(key)) + len(key) + 5  # 3: two double quotes and one colon
        res = response[pos:response.find('"', pos)]
        results[key] = res.replace('"', '').strip()
    return results


use_gui = False

if use_gui:
    # open browser:  http://127.0.0.1:7860
    iface = gr.Interface(fn=rag_chain,
                         inputs=["text", "text"],
                         outputs="text")
    iface.launch()
else:

    queue = [
        'I need to go to the kitchen',
        'I need to take the bottle',
        'what do you see around?'
    ]


    while True:
        # prompt = input("Type your question ('bye' to exit): ")
        # if prompt == 'bye':
        #     break
        if len(queue) == 0:
            break
        prompt = queue.pop()
        print('prompt: ', prompt)
        start = time.time()
        response = rag_chain(filename="./document.json",
                             question=template.format(prompt=prompt))
        print('>> ', response, '\n')
        print('time:', time.time() - start)



