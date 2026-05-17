# import os
# from langchain_openai import ChatOpenAI
# from dotenv import load_dotenv
# from transformers import (
#     BlipProcessor,
#     BlipForConditionalGeneration,
#     CLIPModel,
#     CLIPProcessor,
# )

# load_dotenv()

# def get_model():
#     OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

#     api_key = os.getenv("OPENAI_API_KEY")

#     if not api_key:
#         raise ValueError(
#             "OPENAI_API_KEY is not set. Add it to your .env file."
#         )

#     llm = ChatOpenAI(
#         model=OPENAI_MODEL,
#         temperature=0.2,
#         api_key=api_key
#     )

#     return llm

# def get_blip_captioner(blip_processor=None, blip_model=None):
#     if blip_processor is not None and blip_model is not None:
#         return blip_processor, blip_model

#     # self-contained fallback
#     blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
#     blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
#     return blip_processor, blip_model

# def get_clip_model(clip_processor=None, clip_model=None):
#     if clip_processor is not None and clip_model is not None:
#         return clip_processor, clip_model

#     # self-contained fallback
#     clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
#     clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
#     clip_model.eval()

#     return clip_processor, clip_model

"""
src/llm_setup.py
LLM setup with both OpenAI (local) and AWS Bedrock (production)
"""

import os
import boto3
import json
import logging
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    CLIPModel,
    CLIPProcessor,
)

load_dotenv()

logger = logging.getLogger(__name__)

# ========================================
# ENVIRONMENT VARIABLES
# ========================================

USE_BEDROCK = os.getenv("USE_BEDROCK", "false").lower() == "true"
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
BEDROCK_MODEL_ID = os.getenv(
    "BEDROCK_MODEL_ID",
    "us.anthropic.claude-haiku-4-5-20251001-v1:0"
)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


# ========================================
# LOCAL LLM (OPENAI) - Your current setup
# ========================================

def get_model():
    """Get LLM - OpenAI for local development"""
    
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


# ========================================
# AWS BEDROCK LLM - For production/AWS
# ========================================

class BedrockLLM:
    """Wrapper for AWS Bedrock Claude 3"""
    
    def __init__(self, model_id=None, region=None):
        self.model_id = model_id or BEDROCK_MODEL_ID
        self.region = region or AWS_REGION
        self.client = boto3.client('bedrock-runtime', region_name=self.region)
        logger.info(f"✓ Initialized Bedrock with model: {self.model_id}")
    
    def invoke(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1024):
        """
        Invoke Claude 4.5 via Bedrock
        Compatible with langchain ChatOpenAI interface
        """
        try:
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-06-01",
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ]
                })
            )
            
            result = json.loads(response['body'].read())
            response_text = result['content'][0]['text']
            
            # Return object compatible with langchain
            class Response:
                def __init__(self, content):
                    self.content = content
            
            return Response(response_text)
        
        except Exception as e:
            logger.error(f"Bedrock invocation failed: {e}")
            raise


def get_bedrock_model(model_id=None, region=None):
    """Get AWS Bedrock LLM"""
    return BedrockLLM(model_id=model_id, region=region)


def get_model_adaptive():
    """Get LLM based on environment - OpenAI local, Bedrock for AWS"""
    if USE_BEDROCK:
        logger.info("Using AWS Bedrock")
        return get_bedrock_model()
    else:
        logger.info("Using OpenAI")
        return get_model()


# ========================================
# IMAGE MODELS - Your existing setup
# ========================================

def get_blip_captioner(blip_processor=None, blip_model=None):
    """Get BLIP image captioning model"""
    if blip_processor is not None and blip_model is not None:
        return blip_processor, blip_model

    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return blip_processor, blip_model


def get_clip_model(clip_processor=None, clip_model=None):
    """Get CLIP model for image embeddings"""
    if clip_processor is not None and clip_model is not None:
        return clip_processor, clip_model

    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.eval()

    return clip_processor, clip_model