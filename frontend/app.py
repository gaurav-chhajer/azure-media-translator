import streamlit as st
import requests
import time
import os
import azure.cognitiveservices.speech as speechsdk
import queue 

# --- Imports for WebRTC ---
from streamlit_webrtc import (
    WebRtcMode,
    webrtc_streamer,
    AudioProcessorFactory
)
import av 
import aiortc 

# =============================================================================
# --- (NEW) GET ALL SECRETS ---
# =============================================================================
# For Railway Backend
BACKEND_URL = os.environ.get("BACKEND_URL") 
# For Local Real-Time Mic
AZURE_KEY = os.environ.get("AZURE_KEY")
AZURE_LOCATION = os.environ.get("AZURE_LOCATION")

if not BACKEND_URL or not AZURE_KEY or not AZURE_LOCATION:
    st.error("FATAL: Missing one or more environment variables (BACKEND_URL, AZURE_KEY, AZURE_LOCATION).")
    st.stop()

# =============================================================================
# 1. PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Azure Media Translator",
    page_icon="🎬",
    layout="wide"
)

# =============================================================================
# 2. LANGUAGE AND VOICE CONFIG
# =============================================================================
LANG_OPTIONS = {
    'English': 'en-US', 'Hindi': 'hi-IN', 'French': 'fr-FR', 'Spanish': 'es-ES', 'German': 'de-DE',
    'Tamil': 'ta-IN', 'Telugu': 'te-IN', 'Malayalam': 'ml-IN', 'Arabic': 'ar-AE',
    'Chinese (Simplified)': 'zh-CN', 'Russian': 'ru-RU', 'Japanese': 'ja-JP', 'Korean': 'ko-KR'
}
TTS_VOICE_MAP = {
    'en': 'en-US-JennyNeural', 'hi': 'hi-IN-SwaraNeural', 'fr': 'fr-FR-DeniseNeural',
    'es': 'es-ES-ElviraNeural', 'de': 'de-DE-KatjaNeural', 'ta': 'ta-IN-PallaviNeural',
    'te': 'te-IN-ShrutiNeural', 'ml': 'ml-IN-SobhanaNeural', 'ar': 'ar-EG-SalmaNeural',
    'zh-cn': 'zh-CN-XiaoxiaoNeural',
    'ru': 'ru-RU-SvetlanaNeural', 'ja': 'ja-JP-NanamiNeural',
    'ko': 'ko-KR-SunHiNeural'
}
TRANSLATE_OPTIONS_SHORT = { k: v.split('-')[0] for k, v in LANG_OPTIONS.items() }

# =============================================================================
# 3. (NEW) REAL-TIME MICROPHONE CLASSES (Runs on Hugging Face)
# =============================================================================
@st.cache_resource
def get_mic_translation_config(source_lang_code):
    # This uses the AZURE_KEY from the HF environment
    return speechsdk.translation.SpeechTranslationConfig(
        subscription=AZURE_KEY, 
        region=AZURE_LOCATION,
        speech_recognition_language=source_lang_code
    )

class AzureAudioProcessor:
    def __init__(self, translation_config, target_lang_code, text_queue):
        self.translation_config = translation_config
        self.target_lang_code = target_lang_code
        self.push_stream = speechsdk.audio.PushAudioInputStream(
            stream_format=speechsdk.audio.AudioStreamFormat(samples_per_second=16000, bits_per_sample=16, channels=1)
        )
        self.audio_config = speechsdk.audio.AudioConfig(stream=self.push_stream)
        self.translator = None
        self.is_started = False
        self.text_queue = text_queue

    def handle_translation(self, evt):
        if evt.result.reason == speechsdk.ResultReason.TranslatedSpeech:
            rec_text = evt.result.text
            trans_text = evt.result.translations[self.target_lang_code]
            self.text_queue.put(("RECOGNIZED", rec_text))
            self.text_queue.put(("TRANSLATED", trans_text))

    def stop_cb(self, evt):
        print(f"CLOSING on {evt}")
        if self.translator:
            self.translator.stop_continuous_recognition_async()
        self.is_started = False

    def recv(self, frame: av.AudioFrame):
        if not self.is_started:
            self.translator = speechsdk.translation.TranslationRecognizer(
                translation_config=self.translation_config,
                audio_config=self.audio_config
            )
            self.translator.add_target_language(self.target_lang_code)
            self.translator.recognized.connect(self.handle_translation)
            self.translator.session_stopped.connect(self.stop_cb)
            self.translator.canceled.connect(self.stop_cb)
            self.translator.start_continuous_recognition_async()
            self.is_started = True
            print("--- Azure Translator (Real-Time) started ---")
        
        resampled_frame = frame.reformat(format="s16", layout="mono", rate=16000)
        self.push_stream.write(resampled_frame.to_ndarray().tobytes())
        return frame

    def on_ended(self):
        print("--- WebRTC stream ended, cleaning up Azure ---")
        if self.translator:
            self.translator.stop_continuous_recognition_async()
            self.translator = None
        self.push_stream.close()
        self.is_started = False

class AzureAudioProcessorFactory(AudioProcessorFactory):
    def __init__(self, translation_config, target_lang_code, text_queue):
        self.translation_config = translation_config
        self.target_lang_code = target_lang_code
        self.text_queue = text_queue
    
    def __call__(self):
        return AzureAudioProcessor(
            translation_config=self.translation_config,
            target_lang_code=self.target_lang_code,
            text_queue=self.text_queue
        )

# =============================================================================
# 4. STREAMLIT UI
# =============================================================================

# --- Sidebar Controls ---
st.sidebar.header("⚙️ Configuration")
source_lang_name = st.sidebar.selectbox("Translate from:", options=LANG_OPTIONS.keys(), index=0)
source_lang_code = LANG_OPTIONS[source_lang_name]

target_lang_name = st.sidebar.selectbox("Translate to:", options=TRANSLATE_OPTIONS_SHORT.keys(), index=1)
target_lang_code = TRANSLATE_OPTIONS_SHORT[target_lang_name]
target_voice_name = TTS_VOICE_MAP.get(target_lang_code, "en-US-JennyNeural")

CHUNK_DURATION_SEC = st.sidebar.slider(
    "Processing Chunk Size (seconds)", 
    min_value=30, max_value=120, value=60, step=15,
    help="For File/URL processing. Smaller chunks = faster parallel processing."
)

st.sidebar.info("This app translates video, audio, or live speech.")

# --- Main Page Layout ---
st.title("🎬 Azure Media Translator")
st.info(f"Translate media from **{source_lang_name}** to **{target_lang_name}**.")

input_method = st.radio(
    "Choose media source:", 
    ("YouTube URL", "Upload a File", "Microphone (Real-Time)"), 
    horizontal=True,
    key="input_method"
)

job_id = None
payload = {
    "source_lang_code": source_lang_code,
    "target_lang_code": target_lang_code,
    "target_voice_name": target_voice_name,
    "chunk_duration_sec": CHUNK_DURATION_SEC
}

# =============================================================================
# --- (NEW) REAL-TIME MICROPHONE UI ---
# =============================================================================
if input_method == "Microphone (Real-Time)":
    st.info("Start the microphone to translate your speech in real-time. Make sure to grant browser permission.")
    
    translation_config = get_mic_translation_config(source_lang_code)
    
    if "text_queue" not in st.session_state:
        st.session_state.text_queue = queue.Queue()
    
    webrtc_ctx = webrtc_streamer(
        key="mic_translator",
        mode=WebRtcMode.RECVONLY,
        audio_processor_factory=AzureAudioProcessorFactory(
            translation_config=translation_config,
            target_lang_code=target_lang_code,
            text_queue=st.session_state.text_queue
        ),
        media_stream_constraints={"audio": True, "video": False},
        sendback_audio=False,
    )

    st.subheader("Live Translation")
    col_rec, col_trans = st.columns(2)
    with col_rec:
        rec_box = st.empty()
        rec_box.text_area(f"Recognized ({source_lang_name}):", "", height=200, key="rec_text_live")
    with col_trans:
        trans_box = st.empty()
        trans_box.text_area(f"Translated ({target_lang_name}):", "", height=200, key="trans_text_live")

    if 'rec_text_live_buffer' not in st.session_state:
        st.session_state.rec_text_live_buffer = ""
    if 'trans_text_live_buffer' not in st.session_state:
        st.session_state.trans_text_live_buffer = ""

    while webrtc_ctx.state.playing:
        try:
            text_type, text = st.session_state.text_queue.get(timeout=0.1)
            if text_type == "RECOGNIZED":
                st.session_state.rec_text_live_buffer += f" {text}"
                rec_box.text_area(f"Recognized ({source_lang_name}):", st.session_state.rec_text_live_buffer, height=200)
            elif text_type == "TRANSLATED":
                st.session_state.trans_text_live_buffer += f" {text}"
                trans_box.text_area(f"Translated ({target_lang_name}):", st.session_state.trans_text_live_buffer, height=200)
        except queue.Empty:
            pass
        time.sleep(0.1)
    
    st.session_state.rec_text_live_buffer = ""
    st.session_state.trans_text_live_buffer = ""

# =============================================================================
# --- (EXISTING) FILE & URL UI ---
# =============================================================================
else: # This block handles "YouTube URL" and "Upload a File"
    if input_method == "YouTube URL":
        url = st.text_input("Enter YouTube URL:")
        if st.button("Process and Translate"):
            if url:
                payload["url"] = url
                try:
                    response = requests.post(f"{BACKEND_URL}/process-url", data=payload)
                    if response.status_code == 200:
                        job_id = response.json().get("job_id")
                    else:
                        st.error(f"Error submitting job: {response.text}")
                except Exception as e:
                    st.error(f"Failed to connect to backend: {e}")
            else:
                st.warning("Please enter a URL.")

    elif input_method == "Upload a File":
        uploaded_file = st.file_uploader(
            "Choose a video or audio file...", 
            type=['mp4', 'mkv', 'avi', 'mov', 'webm', 'mp3', 'wav', 'm4a', 'ogg', 'flac']
        )
        if st.button("Process and Translate"):
            if uploaded_file:
                files = {'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                try:
                    response = requests.post(f"{BACKEND_URL}/upload", data=payload, files=files)
                    if response.status_code == 200:
                        job_id = response.json().get("job_id")
                    else:
                        st.error(f"Error submitting job: {response.text}")
                except Exception as e:
                    st.error(f"Failed to connect to backend: {e}")
            else:
                st.warning("Please upload a file.")

    st.divider()

    # --- (EXISTING) Job Polling Section ---
    if job_id:
        st.info(f"Job submitted! ID: {job_id}")
        status_placeholder = st.empty()
        result_placeholder = st.empty()
        
        with st.spinner("Processing... This may take several minutes."):
            while True:
                try:
                    status_response = requests.get(f"{BACKEND_URL}/status/{job_id}")
                    if status_response.status_code != 200:
                        status_placeholder.error(f"Error checking status: {status_response.text}")
                        break
                    
                    status_data = status_response.json()
                    
                    if status_data["status"] == "PROGRESS":
                        status_placeholder.info(status_data.get("message", "Processing..."))
                    elif status_data["status"] == "SUCCESS":
                        status_placeholder.success("Processing complete!")
                        result = status_data.get("result", {})
                        
                        result_path = result.get("result_path")
                        if result_path:
                            file_url = f"{BACKEND_URL}/files/{os.path.basename(result_path)}"
                            if result_path.endswith(".mp4"):
                                result_placeholder.video(file_url)
                            else:
                                result_placeholder.audio(file_url)
                        
                        st.subheader("Text Results")
                        col_res1, col_res2 = st.columns(2)
                        with col_res1:
                            st.text_area(f"Recognized ({source_lang_name}):", result.get("recognized_text", ""), height=150)
                        with col_res2:
                            st.text_area(f"Translated ({target_lang_name}):", result.get("translated_text", ""), height=150)
                        
                        break
                    elif status_data["status"] == "FAILURE":
                        status_placeholder.error(f"Job Failed: {status_data.get('error', 'Unknown error')}")
                        break
                    elif status_data["status"] == "PENDING":
                        status_placeholder.info("Job is queued...")
                    
                    time.sleep(5) # Poll every 5 seconds
                    
                except Exception as e:
                    status_placeholder.error(f"Error connecting to backend: {e}")
                    break