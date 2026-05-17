import os
import json
from datetime import datetime

import numpy as np
import faiss
import torch

from transformers import (
    CLIPModel,
    CLIPProcessor,
)

from src.st_embedding import STEmbedding

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.llm_setup import get_model, get_blip_captioner, get_clip_model

def get_image_embedding(img, clip_processor=None, clip_model=None):
    clip_processor, clip_model = get_clip_model(clip_processor, clip_model)

    inputs = clip_processor(images=img, return_tensors="pt")

    with torch.no_grad():
        image_features = clip_model.get_image_features(**inputs)

    return image_features[0].cpu().numpy().astype("float32")

def analyze_image(
    img,
    user_profile,
    llm,
    blip_processor,
    blip_model,
    state,
    save_dir="stores/image"
):
    if not user_profile:
        return "⚠️ User profile is required for body analysis.", state

    if img is None:
        return "⚠️ Please upload an image.", state

    os.makedirs(save_dir, exist_ok=True)

    text_embeddings = STEmbedding()

    images = img if isinstance(img, list) else [img]

    try:
        image_analysis_texts = []
        clip_meta = []

        for idx, single_img in enumerate(images):
            if single_img is None:
                continue

            # 1. Image embeddings
            # Get embedding for the single image
            vec = get_image_embedding(img)  # returns 1D array
            vec = np.asarray(vec, dtype="float32").reshape(1, -1)  # make it 2D for FAISS

            # 2. BLIP caption
            inputs = blip_processor(images=single_img, return_tensors="pt")
            out = blip_model.generate(**inputs)
            caption = blip_processor.decode(out[0], skip_special_tokens=True)

            # 3. Build LLM prompt
            age = user_profile.get("age", "unknown")
            weight = user_profile.get("weight", "unknown")
            height = user_profile.get("height", "unknown")
            health_issues = user_profile.get("health_issues", "none")
            gender = user_profile.get("gender", "unknown")
            fitness_level = user_profile.get("fitness_level", "beginner")

            prompt = f"""
You are a certified fitness coach. A user uploads a body image which appears as: "{caption}".
Please avoid LaTeX formatting.

User Profile:
- Age: {age}
- Height: {height}
- Weight: {weight}
- Health Concerns: {health_issues}
- Gender: {gender}
- Fitness Level: {fitness_level}

Your task:
1. Give a neutral and supportive observation based on the image and profile:
   - Apparent body type
   - Visible fat distribution
   - Muscle tone visibility
   - Posture or alignment issues

2. Suggest beginner, intermediate, and advanced workout progression:
   - Prefer gym machines
   - Include sets, reps, and form cues
   - Mention when to move to the next level
   - Also provide optional home alternatives

3. Offer coaching tips:
   - Nutrition focus
   - Fat loss vs muscle visibility
   - Motivational closing note

Respect medical conditions.
"""

            result = llm.invoke(prompt).content
            if not result:
                result = "⚠️ LLM did not return any content."
            result = result.strip()
            image_analysis_texts.append({"caption": caption, "analysis": result})

            clip_meta.append({
                "idx": idx,
                "caption": caption,
                "user_profile": user_profile,
                "created_at": str(datetime.now())
            })

        if not image_analysis_texts:
            return "⚠️ Could not analyze the image(s). Please try again with a clearer image.", state

        # Build FAISS index if vectors exist
        clip_index_path, clip_meta_path = None, None

        try:
            faiss.normalize_L2(vec)
            clip_index = faiss.IndexFlatIP(vec.shape[1])
            clip_index.add(vec)

            clip_index_path = os.path.join(save_dir, "clip.index")
            clip_meta_path = os.path.join(save_dir, "clip_meta.json")

            faiss.write_index(clip_index, clip_index_path)
            with open(clip_meta_path, "w") as f:
               json.dump(clip_meta, f, indent=2)
        except Exception as e:
            return f"❌ FAISS indexing error: {e}", state

        # Prepare vectorstore
        try:
            combined_texts = [
                f"Image Description: {item['caption']}\n\nBody Analysis:\n{item['analysis']}"
                for item in image_analysis_texts
            ]
            metadatas = []
            for item in image_analysis_texts:
                md = dict(user_profile)
                md["caption"] = item["caption"]
                md["created_at"] = str(datetime.now())
                metadatas.append(md)
            profile_vectorstore = FAISS.from_texts(
                combined_texts,
                embedding=text_embeddings,
                metadatas=metadatas
            )
            profile_vs_path = os.path.join(save_dir, "profile_vectorstore")
            profile_vectorstore.save_local(profile_vs_path)
        except Exception as e:
            return f"❌ Vectorstore save error: {e}", state

        # Process analysis into structured items
        image_analysis_items = []
        for item in image_analysis_texts:
            caption = item["caption"]
            result_text = item["analysis"]
            try:
                analysis_dict = json.loads(result_text)
            except json.JSONDecodeError:
                analysis_dict = {
                    "observation": result_text,
                    "workout_progression": "",
                    "coaching_tips": ""
                }
            image_analysis_items.append({
                "caption": caption,
                "analysis": analysis_dict
            })

        # Update state safely
        image_state = state.setdefault("image", {})
        image_state["uploaded"] = True
        image_state["vectorstore_path"] = profile_vs_path
        image_state["clip_index_path"] = clip_index_path
        image_state["clip_meta_path"] = clip_meta_path
        image_state["items"] = image_analysis_items
        image_state["analysis_text"] = "\n\n---\n\n".join(
            [f"{i['caption']}\n{i['analysis']['observation']}" for i in image_analysis_items]
        )

        return "✅ Uploaded 1 image.", state

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"❌ Unexpected error: {e}", state