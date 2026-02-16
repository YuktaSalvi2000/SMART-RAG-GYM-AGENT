import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

def get_model():
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is not set. Add it to your .env file."
        )

    llm = ChatOpenAI(
        model=OPENAI_MODEL,
        temperature=0.2,
        api_key=api_key
    )

    return llm

def get_blip_captioner(blip_processor=None, blip_model=None):
    if blip_processor is not None and blip_model is not None:
        return blip_processor, blip_model

    # self-contained fallback
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return blip_processor, blip_model