import os
import shutil
import tempfile
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from celery.result import AsyncResult
from tasks import celery_app, process_media_task, yt_dlp

app = FastAPI()

# Mount a directory to serve the final translated files
os.makedirs("backend/static", exist_ok=True)
app.mount("/static", StaticFiles(directory="backend/static"), name="static")

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    source_lang_code: str = Form(...),
    target_lang_code: str = Form(...),
    target_voice_name: str = Form(...),
    chunk_duration_sec: int = Form(...)
):
    try:
        # Save uploaded file to a temporary path
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Check if it's video or audio
        is_video = file.content_type.startswith("video")
        
        # Start the background task
        task = process_media_task.delay(
            file_path, file.filename, source_lang_code, target_lang_code, 
            target_voice_name, chunk_duration_sec, is_video
        )
        return JSONResponse({"job_id": task.id})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process-url")
async def process_url(
    url: str = Form(...),
    source_lang_code: str = Form(...),
    target_lang_code: str = Form(...),
    target_voice_name: str = Form(...),
    chunk_duration_sec: int = Form(...)
):
    try:
        # --- Download from YouTube ---
        temp_dir = tempfile.mkdtemp()
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'outtmpl': os.path.join(temp_dir, 'downloaded.mp4'),
            'merge_output_format': 'mp4',
            'noplaylist': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)

        # Start the background task (always video for YouTube)
        task = process_media_task.delay(
            filename, "youtube_video.mp4", source_lang_code, target_lang_code, 
            target_voice_name, chunk_duration_sec, is_video=True
        )
        return JSONResponse({"job_id": task.id})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    task_result = AsyncResult(job_id, app=celery_app)
    
    if task_result.state == 'PENDING':
        return {"status": "PENDING", "message": "Job is waiting to start."}
    elif task_result.state == 'PROGRESS':
        return {"status": "PROGRESS", "message": task_result.info.get('status', 'Processing...')}
    elif task_result.state == 'SUCCESS':
        result = task_result.get()
        return {"status": "SUCCESS", "result": result}
    elif task_result.state == 'FAILURE':
        return {"status": "FAILURE", "error": str(task_result.info)}
    else:
        return {"status": task_result.state}

@app.get("/files/{filename}")
async def get_file(filename: str):
    file_path = os.path.join("backend/static", filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    raise HTTPException(status_code=404, detail="File not found")