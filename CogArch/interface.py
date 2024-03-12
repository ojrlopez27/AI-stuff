



agent = Agent()

while True:
    user = input('user >> ')
    if user == 'bye':
        break
    agent.run_cognitive_cycle(user)