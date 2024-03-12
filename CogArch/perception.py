

class Perception():
    percepts = None
    llm = None

    def __init__(self):
        percepts = dict()
        llm = LLM.get_instance()


    def perceive(self, input):
        """
        This method is called when the agent perceives its external world.
        The method converts a natural language input into a symbolic representation
        :param input: natural language input
        :return: a dictionary of percepts
        """
        result = llm.infer('template_perception.txt', input)
        print('perception:', result)