# Warning control
import warnings
warnings.filterwarnings('ignore')

from crewai import Agent, Task, Crew, LLM
import os
import litellm


# patch to avoid error on o1-mini: https://github.com/crewAIInc/crewAI/issues/1908
original_completion = litellm.completion
def patched_completion(*args, **kwargs):
    if 'stop' in kwargs:
        kwargs.pop('stop')
    return original_completion(*args, **kwargs)
litellm.completion = patched_completion



from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_KEY_O1")
API_VERSION = os.getenv("AZURE_API_VERSION_O1")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT_O1")
AZURE_OPENAI_LLM_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME_O1")
LLM_MODEL_NAME = 'o1-mini'  # 'aida' is the deployment name for 'gpt-35-turbo'. See this: https://oai.azure.com/resource/deployments?wsid=/subscriptions/e4125a25-14c7-456b-ba60-7efbb248a7e1/resourceGroups/gpt/providers/Microsoft.CognitiveServices/accounts/laerai-gpt&tid=a67ff07f-4a34-47fe-9378-3249c4878cff

# set environment as LiteLLM expects
os.environ['AZURE_API_KEY'] = AZURE_OPENAI_API_KEY
os.environ["AZURE_API_BASE"] = AZURE_OPENAI_ENDPOINT
os.environ["AZURE_API_VERSION"] = API_VERSION

model = f'azure/{LLM_MODEL_NAME}'

planner = Agent(
    role="Content Planner",
    goal="Plan engaging and factually accurate content on {topic}",
    backstory="You're working on planning a blog article "
              "about the topic: {topic}."
              "You collect information that helps the "
              "audience learn something "
              "and make informed decisions. "
              "Your work is the basis for "
              "the Content Writer to write an article on this topic.",
    allow_delegation=False,
	verbose=True,
    llm=LLM(model=model)
)


writer = Agent(
    role="Content Writer",
    goal="Write insightful and factually accurate "
         "opinion piece about the topic: {topic}",
    backstory="You're working on a writing "
              "a new opinion piece about the topic: {topic}. "
              "You base your writing on the work of "
              "the Content Planner, who provides an outline "
              "and relevant context about the topic. "
              "You follow the main objectives and "
              "direction of the outline, "
              "as provide by the Content Planner. "
              "You also provide objective and impartial insights "
              "and back them up with information "
              "provide by the Content Planner. "
              "You acknowledge in your opinion piece "
              "when your statements are opinions "
              "as opposed to objective statements.",
    allow_delegation=False,
    verbose=True,
    llm=LLM(model=model)
)

editor = Agent(
    role="Editor",
    goal="Edit a given blog post to align with "
         "the writing style of the organization. ",
    backstory="You are an editor who receives a blog post "
              "from the Content Writer. "
              "Your goal is to review the blog post "
              "to ensure that it follows journalistic best practices,"
              "provides balanced viewpoints "
              "when providing opinions or assertions, "
              "and also avoids major controversial topics "
              "or opinions when possible.",
    allow_delegation=False,
    verbose=True,
    llm=LLM(model=model)
)

plan = Task(
    description=(
        "1. Prioritize the latest trends, key players, "
            "and noteworthy news on {topic}.\n"
        "2. Identify the target audience, considering "
            "their interests and pain points.\n"
        "3. Develop a detailed content outline including "
            "an introduction, key points, and a call to action.\n"
        "4. Include SEO keywords and relevant data or sources."
    ),
    expected_output="A comprehensive content plan document "
        "with an outline, audience analysis, "
        "SEO keywords, and resources.",
    agent=planner,
)

write = Task(
    description=(
        "1. Use the content plan to craft a compelling "
            "blog post on {topic}.\n"
        "2. Incorporate SEO keywords naturally.\n"
		"3. Sections/Subtitles are properly named "
            "in an engaging manner.\n"
        "4. Ensure the post is structured with an "
            "engaging introduction, insightful body, "
            "and a summarizing conclusion.\n"
        "5. Proofread for grammatical errors and "
            "alignment with the brand's voice.\n"
    ),
    expected_output="A well-written blog post "
        "in markdown format, ready for publication, "
        "each section should have 2 or 3 paragraphs.",
    agent=writer,
)

edit = Task(
    description=("Proofread the given blog post for "
                 "grammatical errors and "
                 "alignment with the brand's voice."),
    expected_output="A well-written blog post in markdown format, "
                    "ready for publication, "
                    "each section should have 2 or 3 paragraphs.",
    agent=editor
)

crew = Crew(
    agents=[planner, writer, editor],
    tasks=[plan, write, edit],
    verbose=True
)

result = crew.kickoff(inputs={"topic": "Artificial Intelligence"})
print(result)


