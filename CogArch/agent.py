



class Agent():
    perception = Perception()
    wm = WM()
    procedural = Procedural()
    semantic = Semantic()
    Episodic = Episodic()

    def run_cognitive_cycle(self, input):
        percepts = perception.perceive(input)
        return percepts


