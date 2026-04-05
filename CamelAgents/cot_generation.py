from camel.agents import ChatAgent
from camel.datagen import CoTDataGenerator
from camel.types import ModelPlatformType, ModelType
from camel.models import ModelFactory
from camel.configs import ChatGPTConfig
from dotenv import load_dotenv
from pydantic import BaseModel


load_dotenv()

model = ModelFactory.create(
        model_platform=ModelPlatformType.AZURE,
        model_type=ModelType.GPT_4O,
        model_config_dict=ChatGPTConfig(temperature=0.8).as_dict(),
)


# Initialize agents
generator_agent = ChatAgent("System message for generator", model=model)
verifier_agent = ChatAgent("System message for verifier", model=model)


# Define golden answers
golden_answers = {
    "why is the sky blue?": "because the blue color of the ocean reflects on the sky", # wrong
    "(3.24 * 5.34987) / 45.982 + 98.4124 / 2.6791 - 5.2987": "31.811645"
}

# Create generator
cot_generator = CoTDataGenerator(
    generator_agent=generator_agent,
    verifier_agent=verifier_agent,
    golden_answers=golden_answers,
    search_limit=10
)

# Generate solution
# solution = cot_generator.solve("(3.24 * 5.34987) / 45.982 + 98.4124 / 2.6791 - 5.2987")
# cot_generator.export_solutions("solutions_cot_generation.json")
# print(solution)
similarity = cot_generator.monte_carlo_tree_search("why is the sky blue?", "because the water is reflected into the sky")
print(similarity)
