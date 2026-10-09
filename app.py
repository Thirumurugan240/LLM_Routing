import os
import base64
import requests
import streamlit as st
from crewai import Agent, Task, Crew, LLM
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://api.deeprelay.ai/v1"
API_KEY = os.getenv("DEEPRELAY_API_KEY")

MODELS = [
    "deeprelay/qwen3.8-flash",       # Qwen       - fast, cheap
    "deeprelay/deepseek-v4-flash",   # DeepSeek   - fast, cheap
    "deeprelay/gpt-oss-120b",        # OpenAI     - good all rounder
    "deeprelay/glm-5.3",             # Z.AI       - strong
    "deeprelay/deepseek-v4-pro",     # DeepSeek   - strongest, for super tasks
]

IMAGE_MODELS = [
    "deeprelay/flux.2-pro",          # Black Forest Labs
    "deeprelay/gpt-image-1.5",       # OpenAI
    "deeprelay/seedream-4.0",        # ByteDance
    "deeprelay/qwen-image-2.0",      # Qwen
    "deeprelay/flash-image-2.5",     # Google
]

SUPER_TASK_WORDS = ["analyze", "analyse", "plan", "strategy", "code", "research",
                    "compare", "design", "architecture", "detailed", "step by step"]

IMAGE_TASK_WORDS = ["image", "picture", "photo", "draw", "logo", "illustration",
                    "poster", "thumbnail", "wallpaper", "painting"]

st.title("Smart Agent with Auto Model Switching")
st.caption("CrewAI agent + DeepRelay unified API (5 text models, 5 image models)")

question = st.text_area("What do you want the agent to do?")

if st.button("Run Agent") and question:

    is_image_task = any(w in question.lower() for w in IMAGE_TASK_WORDS)

    if is_image_task:
        st.info("Image task detected, switching to image generation models")

        image = None
        for model in IMAGE_MODELS:
            st.write(f"Trying image model: `{model}`")
            try:
                response = requests.post(
                    BASE_URL + "/images/generations",
                    headers={"Authorization": f"Bearer {API_KEY}"},
                    json={"model": model, "prompt": question, "size": "1024x1024"},
                    timeout=180,
                )
                response.raise_for_status()
                image = base64.b64decode(response.json()["data"][0]["b64_json"])
                st.success(f"Image created by `{model}`")
                break

            except Exception as e:
                st.warning(f"`{model}` failed: {e}. Switching to next model...")

        if image:
            st.image(image)
            st.download_button("Download image", image, file_name="image.png")
        else:
            st.error("All image models failed. Check your DEEPRELAY_API_KEY.")

    else:
        is_super_task = len(question.split()) > 40 or any(w in question.lower() for w in SUPER_TASK_WORDS)
        if is_super_task:
            model_order = MODELS[::-1]
            st.info("Super task detected, starting with the strongest model")
        else:
            model_order = MODELS
            st.info("Simple task detected, starting with the fastest model")

        answer = None
        for model in model_order:
            st.write(f"Trying model: `{model}`")
            try:
                llm = LLM(model="openai/" + model, base_url=BASE_URL, api_key=API_KEY)

                agent = Agent(
                    role="Helpful Assistant",
                    goal="Give clear, correct and useful answers",
                    backstory="You are an expert assistant who explains things simply.",
                    llm=llm,
                )

                task = Task(
                    description=question,
                    expected_output="A clear and complete answer",
                    agent=agent,
                )

                crew = Crew(agents=[agent], tasks=[task])
                answer = crew.kickoff()
                st.success(f"Answered by `{model}`")
                break

            except Exception as e:
                st.warning(f"`{model}` failed: {e}. Switching to next model...")

        if answer:
            st.markdown(answer.raw)
        else:
            st.error("All models failed. Check your DEEPRELAY_API_KEY.")
