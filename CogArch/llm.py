import ollama

import utils


class LLM():

    def infer(self, template_path, prompt):
        template = utils.load_txt(template_path, folder='templates/')
        result = ollama.generate('mistral:instruct',
                                 prompt=template.format(prompt=prompt),
                                 options={
                                     'num_gpu': 32,
                                     'main_gpu': 0,
                                     'num_thread': 48,
                                     'temperature': 0.0,
                                     'seed': 42,
                                     # 'top_k': 10,
                                     # 'top_p': 0.5,
                                     'stop': ['</response>']
                                 })

        result = result['response']
        return result