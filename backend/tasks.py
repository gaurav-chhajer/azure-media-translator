import os
import tempfile
import shutil
import concurrent.futures
import numpy as np
import yt_dlp
import moviepy.editor as mp
import azure.cognitiveservices.speech as speechsdk
from celery import Celery

# --- Get secrets from environment ---
AZURE_KEY = os.environ.get("AZURE_KEY")
AZURE_LOCATION = os.environ.get("AZURE_LOCATION")
REDIS_URL = os.environ.get("REDIS_URL") # Railway provides this

if not all([AZURE_KEY, AZURE_LOCATION, REDIS_URL]):
    raise ValueError("Missing environment variables (AZURE_KEY, AZURE_LOCATION, REDIS_URL)")

# --- Celery Setup ---
celery_app = Celery("tasks", broker=REDIS_URL, backend=REDIS_URL)

# --- Re-usable Helper Functions (Your existing code) ---

def get_azure_configs(source_lang_code, target_voice_name):
    translation_config = speechsdk.translation.SpeechTranslationConfig(
        subscription=AZURE_KEY, 
        region=AZURE_LOCATION
    )
    translation_config.speech_recognition_language = source_lang_code 
    synthesis_config = speechsdk.SpeechConfig(
        subscription=AZURE_KEY, 
        region=AZURE_LOCATION
    )
    synthesis_config.speech_synthesis_voice_name = target_voice_name
    return translation_config, synthesis_config

def run_recognition_pipeline(audio_file_path, target_lang_code, translation_config):
    translator = None
    audio_input = None
    all_segments = []
    recognized_text_fragments = []
    translated_text_fragments = []
    try:
        audio_input = speechsdk.AudioConfig(filename=audio_file_path)
        translator = speechsdk.translation.TranslationRecognizer(
            translation_config=translation_config,
            audio_config=audio_input
        )
        translator.add_target_language(target_lang_code)
        done = threading.Event()
        def handle_translation(evt):
            if evt.result.reason == speechsdk.ResultReason.TranslatedSpeech:
                all_segments.append({
                    'start': evt.result.offset / 10_000_000,
                    'duration': evt.result.duration / 10_000_000,
                    'original': evt.result.text,
                    'translated': evt.result.translations[target_lang_code]
                })
                recognized_text_fragments.append(evt.result.text)
                translated_text_fragments.append(evt.result.translations[target_lang_code])
        def handle_cancellation(evt): done.set()
        def handle_session_stopped(evt): done.set()
        translator.recognized.connect(handle_translation)
        translator.session_stopped.connect(handle_session_stopped)
        translator.canceled.connect(handle_cancellation)
        translator.start_continuous_recognition_async()
        done.wait()
    finally:
        if translator:
            translator.stop_continuous_recognition_async()
            translator.recognized.disconnect_all()
            translator.session_stopped.disconnect_all()
            translator.canceled.disconnect_all()
    return all_segments, " ".join(recognized_text_fragments), " ".join(translated_text_fragments)

def recognize_chunk(task_data):
    chunk_path, idx, target_lang_code, translation_config = task_data
    segments, rec_text, trans_text = run_recognition_pipeline(
        chunk_path, target_lang_code, translation_config
    )
    return (idx, segments, rec_text, trans_text)

def split_audio(audio_path, chunk_duration_sec, temp_dir):
    audio = mp.AudioFileClip(audio_path)
    total_duration = audio.duration
    chunk_paths = []
    for i, start_time in enumerate(range(0, int(total_duration) + 1, chunk_duration_sec)):
        end_time = min(start_time + chunk_duration_sec, total_duration)
        if start_time >= end_time: continue
        chunk_path = os.path.join(temp_dir, f"chunk_{i}.wav")
        try:
            sub_audio = audio.subclip(start_time, end_time)
            sub_audio.write_audiofile(chunk_path, codec='pcm_s16le', fps=16000, nbytes=2, ffmpeg_params=["-ac", "1"], logger=None)
            chunk_paths.append((chunk_path, i, start_time))
        except Exception as e:
            print(f"Skipping audio chunk {i}: {e}")
    return chunk_paths, total_duration

def synthesize_segment(segment_data):
    i, segment, synthesis_config, temp_dir = segment_data
    segment_text = segment['translated']
    segment_start = segment['start']
    if not segment_text or segment_text.isspace(): return None
    segment_audio_path = os.path.join(temp_dir, f"segment_{i}.wav")
    try:
        audio_config_tts = speechsdk.audio.AudioOutputConfig(filename=segment_audio_path)
        synthesizer = speechsdk.SpeechSynthesizer(speech_config=synthesis_config, audio_config=audio_config_tts)
        result_tts = synthesizer.speak_text_async(segment_text).get()
        if result_tts.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            if os.path.exists(segment_audio_path) and os.path.getsize(segment_audio_path) > 44:
                audio_clip = mp.AudioFileClip(segment_audio_path)
                clip_array = audio_clip.to_soundarray(fps=16000) 
                return (segment_start, clip_array)
    except Exception as e:
        print(f"Error in synthesis thread: {e}")
    return None

def mix_audio(segment_arrays, total_duration, is_video=True):
    fps = 16000
    num_samples = int(total_duration * fps)
    channels = 2 if is_video else 1
    final_audio_array = np.zeros((num_samples, channels), dtype=np.float32)

    for start_time, clip_array in segment_arrays:
        start_sample = int(start_time * fps)
        end_sample = start_sample + len(clip_array)
        if end_sample > num_samples:
            clip_array = clip_array[:num_samples - start_sample]
        
        if channels == 2 and clip_array.ndim == 1:
            clip_array = np.column_stack([clip_array, clip_array])
        elif channels == 1 and clip_array.ndim == 2:
            clip_array = clip_array[:, 0].reshape(-1, 1)

        final_audio_array[start_sample:end_sample] += clip_array
    
    return mp.AudioArrayClip(final_audio_array, fps=fps)

# --- Main Celery Task ---
@celery_app.task(bind=True)
def process_media_task(self, input_path, original_filename, source_lang_code, target_lang_code, target_voice_name, chunk_duration_sec, is_video):
    
    temp_dir = tempfile.mkdtemp()
    final_output_path = os.path.join("backend/static", f"translated_{original_filename}")
    
    # Ensure static directory exists
    os.makedirs("backend/static", exist_ok=True)
    
    try:
        translation_config, synthesis_config = get_azure_configs(source_lang_code, target_voice_name)
        
        self.update_state(state='PROGRESS', meta={'status': 'Step 1/4: Extracting audio...'})
        if is_video:
            video_clip = mp.VideoFileClip(input_path)
            audio_clip = video_clip.audio
            total_duration = video_clip.duration
        else:
            video_clip = None
            audio_clip = mp.AudioFileClip(input_path)
            total_duration = audio_clip.duration

        temp_audio_path = os.path.join(temp_dir, "original_audio.wav")
        audio_clip.write_audiofile(temp_audio_path, codec='pcm_s16le', fps=16000, nbytes=2, ffmpeg_params=["-ac", "1"], logger=None)
        
        self.update_state(state='PROGRESS', meta={'status': 'Step 2/4: Recognizing chunks...'})
        chunk_data, _ = split_audio(temp_audio_path, chunk_duration_sec, temp_dir)
        tasks = [(path, idx, target_lang_code, translation_config) for (path, idx, start) in chunk_data]
        results = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(recognize_chunk, tasks))
        
        results.sort(key=lambda x: x[0])
        all_segments, rec_text_fragments, trans_text_fragments = [], [], []
        for idx, chunk_segments, rec, trans in results:
            if chunk_segments:
                time_offset = chunk_data[idx][2]
                for segment in chunk_segments:
                    segment['start'] += time_offset
                    all_segments.append(segment)
                rec_text_fragments.append(rec)
                trans_text_fragments.append(trans)
        
        self.update_state(state='PROGRESS', meta={'status': 'Step 3/4: Synthesizing audio...'})
        segment_arrays = []
        if all_segments:
            tasks = [(i, seg, synthesis_config, temp_dir) for i, seg in enumerate(all_segments)]
            with concurrent.futures.ThreadPoolExecutor() as executor:
                results = list(executor.map(synthesize_segment, tasks))
            segment_arrays = [res for res in results if res is not None]

        if not segment_arrays:
            raise Exception("Synthesis failed, no audio segments produced.")

        self.update_state(state='PROGRESS', meta={'status': 'Step 4/4: Mixing final media...'})
        final_audio_clip = mix_audio(segment_arrays, total_duration, is_video)
        
        if is_video:
            final_video_clip = video_clip.set_audio(final_audio_clip)
            final_video_clip.write_videofile(final_output_path, codec='libx264', audio_codec='aac', logger=None)
        else:
            final_audio_clip.write_audiofile(final_output_path, codec='pcm_s16le', fps=16000, logger=None)
        
        return {
            'status': 'SUCCESS',
            'result_path': final_output_path,
            'recognized_text': " ".join(rec_text_fragments),
            'translated_text': " ".join(trans_text_fragments)
        }
        
    except Exception as e:
        return {'status': 'FAILURE', 'error': str(e)}
    finally:
        shutil.rmtree(temp_dir)
        if os.path.exists(input_path):
            os.remove(input_path) # Clean up original upload