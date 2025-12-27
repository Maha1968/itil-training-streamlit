import os
import time
import json
import random
import re
import logging
import base64
from typing import List, Dict, Any, Optional
from datetime import datetime

import streamlit as st
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont
from PyPDF2 import PdfReader


# =====================================================
# BASIC CONFIG
# =====================================================
COMPANY_NAME = "Nexware Technologies Private Limited"
POLICY_NAME = f"{COMPANY_NAME} – ITIL Training"
ALLOWED_EMAIL_DOMAIN = "@nexware-global.com"

MOCK_TEST_QUESTIONS = 40
MOCK_TEST_TIME_MINUTES = 15
MOCK_TEST_PASSING_SCORE = 30

MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# =====================================================
# LOGGING
# =====================================================
logger = logging.getLogger("itil_training")
logger.setLevel(logging.INFO)

_stream_handler = logging.StreamHandler()
_stream_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
_stream_handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(_stream_handler)


def log_event(level: str, msg: str):
    if level.lower() == "error":
        logger.error(msg)
    else:
        logger.info(msg)


# =====================================================
# LOAD POLICY (PDF)
# =====================================================
def load_policy_text(path: str = "itil_trg.pdf") -> str:
    possible_paths = [
        path,
        "itil_trg.pdf",
        os.path.join(os.path.dirname(__file__), "itil_trg.pdf"),
        os.path.join(os.getcwd(), "itil_trg.pdf"),
        os.path.join("assets", "itil_trg.pdf"),
        os.path.join(os.path.dirname(__file__), "assets", "itil_trg.pdf"),
    ]
    
    actual_path = None
    for p in possible_paths:
        if os.path.exists(p):
            actual_path = p
            break
    
    if actual_path is None:
        raise FileNotFoundError(f"itil_trg.pdf not found")
    
    try:
        reader = PdfReader(actual_path)
        all_text = []
        for page in reader.pages:
            text = page.extract_text()
            if text.strip():
                all_text.append(text)
        full_text = "\n\n".join(all_text)
        if not full_text.strip():
            raise ValueError("itil_trg.pdf was found but extracted text is empty")
        return full_text
    except Exception as e:
        raise ValueError(f"Failed to read PDF file: {e}")


# =====================================================
# LOAD TRAINING SLIDES
# =====================================================
def load_training_slides(path: str = "itil_trg.pdf") -> List[Dict[str, Any]]:
    possible_paths = [
        path,
        "itil_trg.pdf",
        os.path.join(os.path.dirname(__file__), "itil_trg.pdf"),
        os.path.join(os.getcwd(), "itil_trg.pdf"),
        os.path.join("assets", "itil_trg.pdf"),
        os.path.join(os.path.dirname(__file__), "assets", "itil_trg.pdf"),
    ]
    
    actual_path = None
    for p in possible_paths:
        if os.path.exists(p):
            actual_path = p
            break
    
    if actual_path is None:
        raise FileNotFoundError(f"itil_trg.pdf not found")
    
    try:
        reader = PdfReader(actual_path)
        total_pages = len(reader.pages)
        
        try:
            first_page = reader.pages[0]
            mediabox = first_page.mediabox
            width = float(mediabox.width)
            height = float(mediabox.height)
            slide_width = int(width * 96 / 72)
            slide_height = int(height * 96 / 72)
        except:
            slide_width = 1920
            slide_height = 1080
        
        aspect_ratio = slide_width / slide_height if slide_height > 0 else 16/9
    except Exception as e:
        raise ValueError(f"Failed to read PDF file: {e}")
    
    slides = []
    for page_idx in range(1, total_pages + 1):
        slides.append({
            "day": 1,
            "slide": page_idx,
            "pdf_path": actual_path,
            "is_separator": False,
            "width": slide_width,
            "height": slide_height,
            "aspect_ratio": aspect_ratio
        })
    
    if not slides:
        raise ValueError("itil_trg.pdf was found but no slides were created")
    
    log_event("info", f"Loaded {len(slides)} slides from PDF file")
    return slides


# =====================================================
# OPENAI CLIENT
# =====================================================
@st.cache_resource
def get_openai_client():
    if OPENAI_API_KEY:
        try:
            return OpenAI(api_key=OPENAI_API_KEY)
        except Exception as e:
            log_event("error", f"OpenAI client init failed: {e}")
            return None
    return None


# =====================================================
# AI CONTENT (LOAD ONCE + CACHE)
# =====================================================
@st.cache_data
def load_training_content():
    """Load PDF content and slides. Cached to prevent reloading."""
    try:
        policy_text = load_policy_text("itil_trg.pdf")
        slides = load_training_slides("itil_trg.pdf")
        log_event("info", f"Training content loaded. {len(slides)} slides")
        return policy_text, slides, None
    except Exception as e:
        error_msg = str(e)
        log_event("error", f"Failed to load training content: {error_msg}")
        return None, None, error_msg


# =====================================================
# MOCK TEST QUESTIONS
# =====================================================
def _safe_json_load(s: str) -> Any:
    s = s.strip()
    if s.startswith("{") or s.startswith("["):
        return json.loads(s)
    i1 = s.find("{")
    i2 = s.find("[")
    i = min([x for x in [i1, i2] if x != -1], default=-1)
    if i == -1:
        raise ValueError("No JSON found in response")
    return json.loads(s[i:])


def generate_mock_test_questions(policy_text: str) -> List[Dict[str, Any]]:
    client = get_openai_client()
    if client is None:
        raise RuntimeError("OpenAI client not available. Check OPENAI_API_KEY.")
    
    log_event("info", f"Generating {MOCK_TEST_QUESTIONS} fresh questions for mock test...")
    
    quiz_prompt = f"""
You are given an ITIL Foundation training content text.
STRICT RULES:
- Use ONLY the content in the text below
- Do NOT add new information
- Questions must be directly answerable from the text
- Create EXACTLY {MOCK_TEST_QUESTIONS} questions
- Make questions diverse and cover different topics from the ITIL Foundation course
OUTPUT FORMAT (JSON ONLY):
[
  {{
    "question": "Question text",
    "options": {{"A":"...", "B":"...", "C":"...", "D":"..."}},
    "correct": "A"
  }}
]
ITIL FOUNDATION TRAINING CONTENT:
{policy_text}
""".strip()

    quiz_resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": quiz_prompt}],
        temperature=0.7,
    )
    quiz_raw = quiz_resp.choices[0].message.content.strip()
    quiz_data = _safe_json_load(quiz_raw)

    if not isinstance(quiz_data, list) or len(quiz_data) < MOCK_TEST_QUESTIONS:
        raise ValueError(f"Quiz JSON invalid or too short. Need exactly {MOCK_TEST_QUESTIONS} questions")

    norm_quiz = []
    for q in quiz_data[:MOCK_TEST_QUESTIONS]:
        question = str(q.get("question", "")).strip()
        options = q.get("options", {})
        correct = str(q.get("correct", "")).strip().upper()

        if not question or not isinstance(options, dict) or correct not in ("A", "B", "C", "D"):
            continue

        for k in ("A", "B", "C", "D"):
            if k not in options:
                options[k] = ""
            options[k] = str(options[k]).strip()

        items = [(k, options[k]) for k in ("A", "B", "C", "D")]
        random.shuffle(items)

        new_options = {}
        new_correct = None
        for idx, (old_k, text) in enumerate(items):
            new_k = ("A", "B", "C", "D")[idx]
            new_options[new_k] = text
            if old_k == correct:
                new_correct = new_k

        if new_correct is None:
            continue

        norm_quiz.append({"question": question, "options": new_options, "correct": new_correct})

    if len(norm_quiz) < MOCK_TEST_QUESTIONS:
        raise ValueError(f"Quiz normalization failed. Need exactly {MOCK_TEST_QUESTIONS}, got {len(norm_quiz)}")
    
    log_event("info", f"Generated {len(norm_quiz)} questions for mock test")
    return norm_quiz


# =====================================================
# CERTIFICATE GENERATION
# =====================================================
def generate_certificate(email: str) -> Optional[str]:
    if not email or '@' not in email:
        return None
    
    name = email.split('@')[0].strip()
    if not name:
        return None
    
    try:
        if not os.path.exists("nexware.png"):
            log_event("error", "nexware.png not found for certificate generation")
            return None
        
        img = Image.open("nexware.png")
        img = img.convert("RGB")
        width, height = img.size
        
        draw = ImageDraw.Draw(img)
        
        font_large = ImageFont.load_default()
        font_medium = ImageFont.load_default()
        font_small = ImageFont.load_default()
        
        try:
            font_paths = [
                "arial.ttf",
                "C:/Windows/Fonts/arial.ttf",
                "C:/Windows/Fonts/Arial.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                "/System/Library/Fonts/Helvetica.ttc"
            ]
            font_path = None
            for path in font_paths:
                if os.path.exists(path):
                    font_path = path
                    break
            
            if font_path:
                font_large = ImageFont.truetype(font_path, 48)
                font_medium = ImageFont.truetype(font_path, 32)
                font_small = ImageFont.truetype(font_path, 24)
        except Exception as e:
            log_event("info", f"Using default font: {e}")
        
        text1 = f"Congratulations {name}"
        text2 = "on successfully completing"
        text3 = "ITIL Training"
        text5 = "Best wishes,"
        text6 = "Nexware Technologies Private Limited"
        text7 = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        
        y_start = height // 3
        line_spacing = 60
        
        bbox1 = draw.textbbox((0, 0), text1, font=font_large)
        text1_width = bbox1[2] - bbox1[0]
        text1_x = (width - text1_width) // 2
        draw.text((text1_x, y_start), text1, fill="black", font=font_large)
        
        bbox2 = draw.textbbox((0, 0), text2, font=font_medium)
        text2_width = bbox2[2] - bbox2[0]
        text2_x = (width - text2_width) // 2
        draw.text((text2_x, y_start + line_spacing), text2, fill="black", font=font_medium)
        
        bbox3 = draw.textbbox((0, 0), text3, font=font_medium)
        text3_width = bbox3[2] - bbox3[0]
        text3_x = (width - text3_width) // 2
        draw.text((text3_x, y_start + line_spacing * 2), text3, fill="black", font=font_medium)
        
        bbox5 = draw.textbbox((0, 0), text5, font=font_medium)
        text5_width = bbox5[2] - bbox5[0]
        text5_x = (width - text5_width) // 2
        draw.text((text5_x, y_start + line_spacing * 4), text5, fill="black", font=font_medium)
        
        bbox6 = draw.textbbox((0, 0), text6, font=font_medium)
        text6_width = bbox6[2] - bbox6[0]
        text6_x = (width - text6_width) // 2
        draw.text((text6_x, y_start + line_spacing * 5), text6, fill="black", font=font_medium)
        
        bbox7 = draw.textbbox((0, 0), text7, font=font_small)
        text7_width = bbox7[2] - bbox7[0]
        text7_x = (width - text7_width) // 2
        draw.text((text7_x, y_start + line_spacing * 7), text7, fill="black", font=font_small)
        
        cert_filename = f"certificate_{name}_{int(time.time())}.png"
        img.save(cert_filename)
        log_event("info", f"Certificate generated: {cert_filename}")
        return cert_filename
        
    except Exception as e:
        log_event("error", f"Certificate generation failed: {e}")
        return None


# =====================================================
# STREAMLIT APP
# =====================================================
st.set_page_config(
    page_title=POLICY_NAME,
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if "training_started" not in st.session_state:
    st.session_state.training_started = False
if "user_email" not in st.session_state:
    st.session_state.user_email = ""
if "slides_loaded" not in st.session_state:
    st.session_state.slides_loaded = False
if "current_slide_idx" not in st.session_state:
    st.session_state.current_slide_idx = 0
if "slides_completed" not in st.session_state:
    st.session_state.slides_completed = False
if "current_step" not in st.session_state:
    st.session_state.current_step = "email_entry"
if "policy_text" not in st.session_state:
    st.session_state.policy_text = None
if "slides" not in st.session_state:
    st.session_state.slides = None
if "load_error" not in st.session_state:
    st.session_state.load_error = None
if "mock_test_questions" not in st.session_state:
    st.session_state.mock_test_questions = []
if "mock_test_idx" not in st.session_state:
    st.session_state.mock_test_idx = 0
if "mock_test_answers" not in st.session_state:
    st.session_state.mock_test_answers = {}
if "mock_test_start_time" not in st.session_state:
    st.session_state.mock_test_start_time = 0.0
if "mock_test_time_up" not in st.session_state:
    st.session_state.mock_test_time_up = False
if "mock_test_submitted" not in st.session_state:
    st.session_state.mock_test_submitted = False
if "mock_test_score" not in st.session_state:
    st.session_state.mock_test_score = 0
if "mock_test_passed" not in st.session_state:
    st.session_state.mock_test_passed = False
if "certificate_file" not in st.session_state:
    st.session_state.certificate_file = None

# Title
st.title(POLICY_NAME)

# =====================================================
# EMAIL ENTRY
# =====================================================
if not st.session_state.training_started:
    st.header("Enter Your Email to Start Training")
    
    email_input = st.text_input(
        "Company Email",
        placeholder=f"you{ALLOWED_EMAIL_DOMAIN}",
        key="email_input"
    )
    
    if st.button("Start Training", type="primary"):
        email = (email_input or "").strip().lower()
        
        if not email:
            st.error("❌ Please enter your email address.")
        elif not email.endswith(ALLOWED_EMAIL_DOMAIN):
            st.error(f"❌ Please use your official {ALLOWED_EMAIL_DOMAIN} email address.")
        else:
            st.session_state.user_email = email
            st.session_state.training_started = True
            st.session_state.current_step = "slides"
            log_event("info", f"Email validated and training started. email={email}")
            st.rerun()

# =====================================================
# TRAINING SLIDES
# =====================================================
elif st.session_state.current_step == "slides":
    # Load slides if not loaded
    if not st.session_state.slides_loaded:
        with st.spinner("⏳ Loading training content... Please wait."):
            policy_text, slides, error = load_training_content()
            if error:
                st.error(f"❌ Training content generation failed: {error}")
                st.session_state.load_error = error
            else:
                st.session_state.policy_text = policy_text
                st.session_state.slides = slides
                st.session_state.slides_loaded = True
                st.rerun()
    
    # Display slides if loaded
    if st.session_state.slides_loaded and st.session_state.slides:
        st.header("1) Training Slides")
        st.success("✅ Training content ready.")
        
        slides = st.session_state.slides
        current_idx = st.session_state.current_slide_idx
        
        # Ensure index is valid
        if current_idx >= len(slides):
            current_idx = 0
            st.session_state.current_slide_idx = 0
        
        current_slide = slides[current_idx]
        slide_num = current_slide["slide"]
        pdf_url = "https://drive.google.com/file/d/1reuk-JLnou4D3rgDbhIA7fhDhqw8RZm9/preview"
        
        # Display PDF page using iframe
        st.components.v1.html(
            f'<iframe src="{pdf_url}#page={slide_num}" width="100%" height="800px" style="border:none;"></iframe>',
            height=820
        )
        
        # Navigation buttons
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("⬅ Previous", disabled=(current_idx == 0)):
                if current_idx > 0:
                    st.session_state.current_slide_idx = current_idx - 1
                    st.rerun()
        
        with col2:
            st.write(f"Slide {slide_num} of {len(slides)}")
        
        with col3:
            if st.button("Next ➡", disabled=(current_idx >= len(slides) - 1)):
                if current_idx < len(slides) - 1:
                    st.session_state.current_slide_idx = current_idx + 1
                    st.rerun()
        
        # Complete Training button (only on last slide)
        is_last_slide = (current_idx == len(slides) - 1)
        if is_last_slide:
            st.info("You have completed all slides. Click **Complete Training** to proceed to the mock test.")
            if st.button("✅ Complete Training", type="primary"):
                st.session_state.slides_completed = True
                st.session_state.current_step = "mock_test"
                log_event("info", "Training slides completed. Moving to mock test.")
                st.rerun()
        else:
            st.info("Use Next/Previous to navigate through the slides.")
    
    elif st.session_state.load_error:
        st.error(f"❌ {st.session_state.load_error}")

# =====================================================
# MOCK TEST
# =====================================================
elif st.session_state.current_step == "mock_test":
    st.header("Final Mock Test")
    st.markdown(f"**{MOCK_TEST_QUESTIONS} questions | {MOCK_TEST_TIME_MINUTES} minutes | Passing score: {MOCK_TEST_PASSING_SCORE}/{MOCK_TEST_QUESTIONS}**")
    
    # Initialize mock test if not started
    if not st.session_state.mock_test_questions and not st.session_state.mock_test_submitted:
        if st.session_state.policy_text:
            with st.spinner("⏳ Generating mock test questions... Please wait."):
                try:
                    questions = generate_mock_test_questions(st.session_state.policy_text)
                    st.session_state.mock_test_questions = questions
                    st.session_state.mock_test_start_time = time.time()
                    st.session_state.mock_test_time_up = False
                    st.session_state.mock_test_idx = 0
                    st.session_state.mock_test_answers = {}
                    log_event("info", f"Mock test started with {len(questions)} questions")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error generating questions: {str(e)}")
                    log_event("error", f"Failed to generate mock test questions: {e}")
        else:
            st.error("❌ Training content not ready yet. Please wait...")
    
    # Display mock test if questions are loaded
    if st.session_state.mock_test_questions and not st.session_state.mock_test_submitted:
        questions = st.session_state.mock_test_questions
        current_q_idx = st.session_state.mock_test_idx
        answers = st.session_state.mock_test_answers
        start_time = st.session_state.mock_test_start_time
        time_up = st.session_state.mock_test_time_up
        
        # Calculate timer
        if start_time > 0:
            elapsed = time.time() - start_time
            total_time = MOCK_TEST_TIME_MINUTES * 60
            remaining = total_time - elapsed
            
            if remaining <= 0 and not time_up:
                st.session_state.mock_test_time_up = True
                time_up = True
                st.warning("⏱️ Time's up!")
                st.rerun()
            
            if not time_up:
                minutes = int(remaining // 60)
                seconds = int(remaining % 60)
                timer_text = f"⏱️ Time remaining: {minutes:02d}:{seconds:02d}"
                progress = min(100.0, (elapsed / total_time) * 100.0)
                
                if progress < 50:
                    color = "#4CAF50"
                elif progress < 80:
                    color = "#FFC107"
                else:
                    color = "#FF9800"
                
                st.markdown(timer_text)
                st.progress(progress / 100.0)
            else:
                timer_text = "⏱️ Time's up!"
                st.markdown(timer_text)
                st.progress(1.0)
        else:
            timer_text = "⏱️ Time remaining: 15:00"
            st.markdown(timer_text)
            st.progress(0.0)
        
        # Auto-submit if time is up
        if time_up and not st.session_state.mock_test_submitted:
            score = 0
            for i, q in enumerate(questions):
                chosen = answers.get(i)
                if chosen and chosen.upper() == q["correct"]:
                    score += 1
            
            st.session_state.mock_test_score = score
            st.session_state.mock_test_passed = score >= MOCK_TEST_PASSING_SCORE
            st.session_state.mock_test_submitted = True
            
            if st.session_state.mock_test_passed:
                cert_file = generate_certificate(st.session_state.user_email)
                if cert_file:
                    st.session_state.certificate_file = cert_file
            
            log_event("info", f"Mock test auto-submitted (time up). score={score}/{MOCK_TEST_QUESTIONS}")
            st.rerun()
        
        # Display current question
        if current_q_idx < len(questions):
            q = questions[current_q_idx]
            st.markdown(f"### Question {current_q_idx + 1}/{len(questions)}")
            st.markdown(f"**{q['question']}**")
            
            # Get current answer if exists
            current_answer = answers.get(current_q_idx, None)
            default_idx = None
            if current_answer:
                options_keys = ['A', 'B', 'C', 'D']
                if current_answer in options_keys:
                    default_idx = options_keys.index(current_answer)
            
            # Radio buttons for options
            options_list = [f"{k}) {q['options'][k]}" for k in ['A', 'B', 'C', 'D']]
            selected_key = st.radio(
                "Select an option",
                options_list,
                index=default_idx,
                key=f"mock_q_{current_q_idx}",
                disabled=time_up
            )
            
            # Save answer
            if selected_key and not time_up:
                choice = selected_key[0]  # Get 'A', 'B', 'C', or 'D'
                st.session_state.mock_test_answers[current_q_idx] = choice
            
            # Navigation buttons
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                if st.button("⬅ Previous", disabled=(current_q_idx == 0 or time_up), key=f"prev_{current_q_idx}"):
                    if current_q_idx > 0:
                        st.session_state.mock_test_idx = current_q_idx - 1
                        st.rerun()
            
            with col2:
                st.write(f"Question {current_q_idx + 1} of {len(questions)}")
            
            with col3:
                if st.button("Next ➡", disabled=(current_q_idx >= len(questions) - 1 or time_up), key=f"next_{current_q_idx}"):
                    if current_q_idx < len(questions) - 1:
                        st.session_state.mock_test_idx = current_q_idx + 1
                        st.rerun()
            
            # Submit button
            if st.button("✅ Submit Mock Test", type="primary", disabled=time_up, key="submit_mock"):
                # Calculate score
                score = 0
                for i, q in enumerate(questions):
                    chosen = answers.get(i)
                    if chosen and chosen.upper() == q["correct"]:
                        score += 1
                
                st.session_state.mock_test_score = score
                st.session_state.mock_test_passed = score >= MOCK_TEST_PASSING_SCORE
                st.session_state.mock_test_submitted = True
                
                if st.session_state.mock_test_passed:
                    # Generate certificate
                    cert_file = generate_certificate(st.session_state.user_email)
                    if cert_file:
                        st.session_state.certificate_file = cert_file
                
                log_event("info", f"Mock test submitted. score={score}/{MOCK_TEST_QUESTIONS}")
                st.rerun()
    
    # Display results if submitted
    if st.session_state.mock_test_submitted:
        score = st.session_state.mock_test_score
        passed = st.session_state.mock_test_passed
        
        if passed:
            st.success(f"✅ Passed! Score: **{score}/{MOCK_TEST_QUESTIONS}** (required: {MOCK_TEST_PASSING_SCORE})")
            if st.session_state.certificate_file:
                st.info("🎉 Certificate generated! Download it below.")
                with open(st.session_state.certificate_file, "rb") as f:
                    st.download_button(
                        "Download Certificate",
                        f.read(),
                        file_name=os.path.basename(st.session_state.certificate_file),
                        mime="image/png"
                    )
            st.session_state.current_step = "done"
            st.rerun()
        else:
            st.error(f"❌ Not passed. Score: **{score}/{MOCK_TEST_QUESTIONS}** (required: {MOCK_TEST_PASSING_SCORE})")
            if st.button("Try Again", type="primary"):
                st.session_state.mock_test_questions = []
                st.session_state.mock_test_submitted = False
                st.session_state.mock_test_idx = 0
                st.session_state.mock_test_answers = {}
                st.session_state.mock_test_start_time = 0.0
                st.session_state.mock_test_time_up = False
                st.rerun()

# =====================================================
# TRAINING COMPLETED
# =====================================================
elif st.session_state.current_step == "done":
    st.header("✅ Training Completed")
    name = st.session_state.user_email.split('@')[0] if st.session_state.user_email else "Employee"
    score = st.session_state.mock_test_score
    st.success(f"**Congratulations {name}**\n\nYou have successfully completed the ITIL Foundation Training\n\n**Score: {score}/{MOCK_TEST_QUESTIONS}**\n\n**{COMPANY_NAME}**\n\n{datetime.now().strftime('%B %d, %Y')}")
    
    if st.session_state.certificate_file and os.path.exists(st.session_state.certificate_file):
        st.info("🎉 Download your certificate below.")
        with open(st.session_state.certificate_file, "rb") as f:
            st.download_button(
                "Download Certificate",
                f.read(),
                file_name=os.path.basename(st.session_state.certificate_file),
                mime="image/png"
            )

# =====================================================
# DEBUG LOGS (Optional - can be removed for production)
# =====================================================
with st.expander("🛠 Debug Logs (for admin)", expanded=False):
    st.text("Logs are available in the console output.")
