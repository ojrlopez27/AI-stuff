import asyncio
import gradio as gr
from google.adk.agents import LlmAgent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types
from dotenv import load_dotenv
from google.adk.models.lite_llm import LiteLlm


load_dotenv()

# Define the agent
agent = LlmAgent(
    name="echo_agent",
    model=LiteLlm(model="azure/gpt-4o-mini"),
    instruction="You are an assistant that answer user's questions."
)

# Initialize session and runner
session_service = InMemorySessionService()
session = session_service.create_session(
    state={},
    app_name="echo_app",
    user_id="user_123"
)
runner = Runner(
    app_name="echo_app",
    agent=agent,
    session_service=session_service
)

# Define the function to process user input through the agent
async def chat_with_agent(user_input):
    content = types.Content(role="user", parts=[types.Part(text=user_input)])
    events = runner.run_async(
        session_id=session.id,
        user_id=session.user_id,
        new_message=content
    )
    response = ""
    async for event in events:
        for part in event.content.parts:
            if hasattr(part, "text"):
                response += part.text
    return response

# Wrap the async function for Gradio
def gradio_chat(user_input):
    return asyncio.run(chat_with_agent(user_input))

# Create the Gradio interface
iface = gr.Interface(
    fn=gradio_chat,
    inputs=gr.Textbox(lines=2, placeholder="Enter your message here..."),
    outputs="text",
    title="ADK + Gradio Chatbot",
    description="A simple chatbot interface using Google's ADK and Gradio."
)

# Launch the Gradio app
if __name__ == "__main__":
    iface.launch()
