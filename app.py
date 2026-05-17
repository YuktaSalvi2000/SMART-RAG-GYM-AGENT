import gradio as gr

from src.llm_setup import get_model, get_blip_captioner
from src.profile_store import create_profile
from src.image_pipeline import analyze_image
from src.pdf_upload import upload_and_index
from backend import init_state, cb_create_profile, cb_analyze, cb_image_chat, cb_pdf_upload, cb_pdf_chat

import numpy as np
import faiss
import os
import json
from datetime import datetime
import torch
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    CSVLoader
)

from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PIL import Image

from transformers import (
    CLIPProcessor,
    CLIPModel,
    BlipProcessor,
    BlipForConditionalGeneration
)

def toggle_mode(mode):
    return (
        gr.update(visible=(mode == "Image Analyzer")),
        gr.update(visible=(mode == "PDF Reader")),
    )

llm = get_model()

with gr.Blocks() as demo:
    gr.Markdown("# 🏋️ Smart Gym RAG (Image Analyzer Lite)")
    gr.Markdown("Free HF Space: caption-based image analysis + PDF reader + chat memory via FAISS.")

    state = gr.State(init_state())

    # ------------------ STEP 1: PROFILE (VISIBLE INITIALLY) ------------------ #
    gr.Markdown("## 1) Create Profile")
    with gr.Row():
        age = gr.Number(label="Age", value=25)
        height = gr.Number(label="Height (cm)", value=160)
        weight = gr.Number(label="Weight (kg)", value=65)

    with gr.Row():
        fitness_level = gr.Dropdown(["beginner", "intermediate", "advanced"], value="beginner", label="Fitness level")
        gender = gr.Radio(["male", "female"], value="female", label="Gender")
        health_issues = gr.Textbox(label="Health issues", value="None")

    btn_profile = gr.Button("Create Profile")
    profile_status = gr.Textbox(label="Status", lines=2)

    btn_profile.click(
        cb_create_profile,
        inputs=[age, height, weight, fitness_level, gender, health_issues, state],
        outputs=[profile_status, state],
    )

    # ------------------ STEP 2: MODE SELECTOR (HIDDEN INITIALLY) ------------------ #
    mode_selector = gr.Dropdown(
        choices=["Image Analyzer", "PDF Reader"],
        label="Choose Mode",
        visible=False
    )

    # Show mode selector after profile created
    def show_mode_selector(status_text, state):
        # status_text is unused; just use it to chain
        return gr.update(visible=True)

    btn_profile.click(
        show_mode_selector,
        inputs=[profile_status, state],
        outputs=[mode_selector],
    )

    # ------------------ IMAGE ANALYZER SECTION (HIDDEN INITIALLY) ------------------ #
    with gr.Column(visible=False) as image_section:
        gr.Markdown("## 2A) Upload Image + Analyze")
        img = gr.Image(type="pil", label="Upload image")
        btn_analyze = gr.Button("Analyze Image")
        analysis_out = gr.Textbox(label="Image Analysis Output", lines=12)

        image_chatbot = gr.Chatbot(label="Chat about your Image Analysis")
        image_msg = gr.Textbox(label="Ask a question about your body / plan")
        btn_image_send = gr.Button("Send")

        btn_analyze.click(
            cb_analyze,
            inputs=[img, state],
            outputs=[analysis_out, state],
        )

        btn_image_send.click(
            cb_image_chat,
            inputs=[image_msg, image_chatbot, state],
            outputs=[image_chatbot, image_msg, state],
        )

    # ------------------ PDF READER SECTION (HIDDEN INITIALLY) ------------------ #
    with gr.Column(visible=False) as pdf_section:
        gr.Markdown("## 2B) Upload PDF/DOCX/CSV + Chat (PDF Reader)")

        pdf_files = gr.File(
            file_types=[".pdf", ".docx", ".csv"],
            file_count="multiple",
            label="Upload your workout/diet plan files"
        )
        btn_pdf_process = gr.Button("Process Files")
        pdf_status = gr.Textbox(label="PDF Upload Status", lines=2)

        pdf_chatbot = gr.Chatbot(label="Chat about your PDF Plan")
        pdf_msg = gr.Textbox(label="Ask a question about your uploaded plan")
        btn_pdf_send = gr.Button("Send (PDF Chat)")

        btn_pdf_process.click(
            cb_pdf_upload,
            inputs=[pdf_files, state],
            outputs=[pdf_status, state],
        )

        btn_pdf_send.click(
            cb_pdf_chat,
            inputs=[pdf_msg, pdf_chatbot, state],
            outputs=[pdf_chatbot, pdf_msg, state],
        )

    # ------------------ MODE TOGGLE WIRING ------------------ #
    mode_selector.change(
        toggle_mode,
        inputs=[mode_selector],
        outputs=[image_section, pdf_section]
    )
demo.launch(
    server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
    server_port=int(os.getenv("GRADIO_SERVER_PORT", 7860)))