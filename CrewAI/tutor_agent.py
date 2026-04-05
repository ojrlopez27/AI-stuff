import warnings
warnings.filterwarnings('ignore')

from crewai import Agent, Task, Crew, LLM, Process
import os
from dotenv import load_dotenv
from crewai_tools import SerperDevTool

load_dotenv()

# Using the env vars from the .env file to set up the model
# Check .env file, variables for Azure
model = f'azure/{os.environ["LLM_MODEL_NAME"]}'

search_tool = SerperDevTool()

expert_tutor_agent = Agent(
    role="Expert Science Teacher for Elementary School",
    goal="Teach complex scientific concepts to elementary "
         "school students in such a way that is engaging "
         "and easy to understand. ",
    backstory=(
        "As an expert science teacher, your mission is "
        "to explain complex scientific concepts of a "
        "specific topic to kids between 6 and 8 years "
        "old. You have been recognized as the best teacher "
        "to be able to break down difficult ideas into "
        "simple, relatable terms that young students can "
        "grasp. You use analogies, stories, and interactive "
        "activities to make learning fun and memorable. "
    ),
    allow_delegation=False,
    verbose=True,
    llm=LLM(model=model)
)

teaching_task = Task(
    description=(
        "Given the science topic of {topic}, you will tell a "
        "story that explains the concept in a way that is easy "
        "to grasp. You can invent the characters and situations "
        "to illustrate the concept to student named {name}. Make "
        "sure to include all the key concepts that are relevant to "
        "the topic. The story should be engaging and fun. Then, "
        "translate it into {language}."
    ),
    expected_output=(
        "Two engaging stories: the first is written in English "
        "and must be engaging and fun for a child named {name} "
        "to understand. The second is the same story translated "
        "into {language}. The story should include all the "
        "key concepts relevant to {topic}."
    ),
    tools=[search_tool],
    agent=expert_tutor_agent,
)

crew = Crew(
    agents=[expert_tutor_agent],
    tasks=[teaching_task],
    verbose=True,
    process=Process.sequential,
    memory=True,
    embedder={
        "provider": "ollama",
        "config": {
            "model": "nomic-embed-text"
        }
    }
)

inputs = {
    "name": "Nicolas",
    "topic": "Chip Majorana",
    "language": "Spanish"
}

result = crew.kickoff(inputs=inputs)

print(result.raw)