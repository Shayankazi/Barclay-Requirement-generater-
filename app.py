from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import asyncio
import os
import sys
import json
import threading
import logging
import subprocess
import traceback
from typing import Dict, List, Set
from requirement_analyzer import RequirementsAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# Ensure static directory exists
os.makedirs("static", exist_ok=True)

class AnalyzerManager:
    def __init__(self):
        self.message_queues = {}
        self.active_websockets = {}
        self.input_events = {}
        self.input_responses = {}
        self.lock = threading.Lock()

    async def send_message(self, websocket_id: str, message: dict):
        if websocket_id in self.active_websockets:
            try:
                await self.active_websockets[websocket_id].send_json(message)
            except Exception as e:
                logger.error(f"Error sending message: {e}")

    def register_websocket(self, websocket_id: str, websocket: WebSocket):
        with self.lock:
            self.message_queues[websocket_id] = []
            self.active_websockets[websocket_id] = websocket
            self.input_events[websocket_id] = threading.Event()
            self.input_responses[websocket_id] = None

    def unregister_websocket(self, websocket_id: str):
        with self.lock:
            self.message_queues.pop(websocket_id, None)
            self.active_websockets.pop(websocket_id, None)
            if websocket_id in self.input_events:
                self.input_events[websocket_id].set()
            self.input_events.pop(websocket_id, None)
            self.input_responses.pop(websocket_id, None)

    def get_queue(self, websocket_id: str) -> list:
        return self.message_queues.get(websocket_id)

    def wait_for_input(self, websocket_id: str, timeout=None) -> str:
        if self.input_events[websocket_id].wait(timeout):
            response = self.input_responses[websocket_id]
            self.input_responses[websocket_id] = None
            self.input_events[websocket_id].clear()
            return response
        return None

    def set_input_response(self, websocket_id: str, response: str):
        with self.lock:
            self.input_responses[websocket_id] = response
            self.input_events[websocket_id].set()

analyzer_manager = AnalyzerManager()

@app.get("/")
async def read_root():
    return FileResponse("static/index.html")

@app.get("/download")
async def download_requirements():
    try:
        if not os.path.exists("requirements_answers.txt"):
            raise HTTPException(status_code=404, detail="File not found")
        return FileResponse(
            path="requirements_answers.txt",
            filename="requirements_answers.txt",
            media_type="text/plain"
        )
    except Exception as e:
        logger.error(f"Download error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/convert_srs")
async def convert_srs(format: str):
    """Convert SRS to different formats"""
    try:
        if format == "word":
            subprocess.run(["pandoc", "system_srs.md", "-o", "system_srs.docx"], check=True)
            return JSONResponse({"success": True})
        elif format == "pdf":
            subprocess.run(["pandoc", "system_srs.md", "-o", "system_srs.pdf"], check=True)
            return JSONResponse({"success": True})
        else:
            raise HTTPException(status_code=400, detail="Invalid format")
    except Exception as e:
        logger.error(f"Error converting SRS: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download_srs")
async def download_srs(format: str = None):
    """Download SRS in various formats"""
    try:
        if not format or format == "md":
            if not os.path.exists("system_srs.md"):
                raise HTTPException(status_code=404, detail="SRS file not found")
            return FileResponse(
                path="system_srs.md",
                filename="system_srs.md",
                media_type="text/markdown"
            )
        elif format == "word":
            if not os.path.exists("system_srs.docx"):
                raise HTTPException(status_code=404, detail="Word file not found")
            return FileResponse("system_srs.docx", filename="system_srs.docx")
        elif format == "pdf":
            if not os.path.exists("system_srs.pdf"):
                raise HTTPException(status_code=404, detail="PDF file not found")
            return FileResponse("system_srs.pdf", filename="system_srs.pdf")
    except Exception as e:
        logger.error(f"Download SRS error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/create_jira_stories")
async def create_jira_stories():
    """Create Jira stories from SRS"""
    try:
        # Check if system_srs.md exists
        if not os.path.exists("system_srs.md"):
            raise HTTPException(status_code=400, detail="No SRS file found. Please generate SRS first.")
        
        # Read the SRS file
        with open("system_srs.md", "r") as f:
            srs_content = f.read()
        
        # Save as txt for Jira processing
        txt_path = "system_srs.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(srs_content)
        
        # Create a StringIO to capture output
        from io import StringIO
        import sys
        output_capture = StringIO()
        original_stdout = sys.stdout
        sys.stdout = output_capture
        
        try:
            # Import main.py functions
            from main import process_srs_file
            
            # Process the file
            process_srs_file(txt_path)
            
            # Get the captured output
            output = output_capture.getvalue()
            print(f"Processing output: {output}")  # Debug line
        finally:
            sys.stdout = original_stdout
            # Clean up the temporary txt file
            if os.path.exists(txt_path):
                os.remove(txt_path)
        
        return JSONResponse({
            "success": True,
            "output": output
        })
    except Exception as e:
        logger.error(f"Error in Jira integration: {e}")
        return JSONResponse({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        })

@app.get("/download_excel")
async def download_excel():
    """Download the generated Excel file from Jira requirements"""
    try:
        if not os.path.exists("requirements.xlsx"):
            raise HTTPException(status_code=404, detail="Excel file not found")
        return FileResponse("requirements.xlsx", filename="requirements.xlsx")
    except Exception as e:
        logger.error(f"Download Excel error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def process_analyzer_queue(websocket_id: str):
    queue = analyzer_manager.get_queue(websocket_id)
    while True:
        try:
            message = queue.pop(0)
            await analyzer_manager.send_message(websocket_id, message)
        except asyncio.CancelledError:
            break
        except Exception:
            await asyncio.sleep(0.1)

def run_analyzer_in_thread(analyzer: RequirementsAnalyzer, project_description: str, websocket_id: str):
    queue = analyzer_manager.get_queue(websocket_id)
    
    def custom_print(*args, **kwargs):
        message = " ".join(str(arg) for arg in args)
        queue.append({
            "type": "output",
            "message": message
        })

    def custom_input(*args, **kwargs):
        prompt = args[0] if args else "Your answer:"
        queue.append({
            "type": "question",
            "message": prompt
        })
        
        response = analyzer_manager.wait_for_input(websocket_id)
        return response

    try:
        # Replace standard input/output
        import builtins
        original_input = builtins.input
        original_print = builtins.print
        builtins.input = custom_input
        builtins.print = custom_print

        # Run analysis
        results = analyzer.analyze_requirements(project_description)
        
        # Save results to file
        try:
            # First clear the existing file
            with open("requirements_answers.txt", 'w', encoding='utf-8') as f:
                f.write("")  # Clear file
                
            # Then save new results
            analyzer.save_requirements("requirements_answers.txt", results)
            
            queue.append({
                "type": "output",
                "message": "\nRequirements have been saved to 'requirements_answers.txt'"
            })
            queue.append({
                "type": "file_ready",
                "message": "requirements_answers.txt"
            })
        except Exception as e:
            logger.error(f"Error saving file: {str(e)}")
            queue.append({
                "type": "error",
                "message": f"Error saving file: {str(e)}"
            })
            
    except Exception as e:
        logger.error(f"Error in analyzer thread: {str(e)}")
        queue.append({
            "type": "error",
            "message": f"Error during analysis: {str(e)}"
        })
    finally:
        # Restore original input/output
        builtins.input = original_input
        builtins.print = original_print

def run_srs_generator_in_thread(websocket_id: str):
    queue = analyzer_manager.get_queue(websocket_id)
    
    try:
        # Check if requirements file exists and is not empty
        if not os.path.exists("requirements_answers.txt") or os.path.getsize("requirements_answers.txt") == 0:
            queue.append({
                "type": "error",
                "message": "No requirements analysis results found. Please analyze requirements first."
            })
            return
            
        # Run srs_generator_v2.py
        process = subprocess.Popen(
            ["python", "srs_generator_v2.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        # Stream output in real-time
        while True:
            output = process.stdout.readline()
            if output:
                queue.append({
                    "type": "output",
                    "message": output.strip()
                })
            
            error = process.stderr.readline()
            if error:
                queue.append({
                    "type": "output",
                    "message": "Error: " + error.strip()
                })
            
            if output == '' and error == '' and process.poll() is not None:
                break

        if process.returncode == 0:
            # Check if SRS file was generated
            if os.path.exists("system_srs.md"):
                queue.append({
                    "type": "output",
                    "message": "\nSRS document has been generated successfully!"
                })
                queue.append({
                    "type": "file_ready",
                    "message": "system_srs.md"
                })
            else:
                queue.append({
                    "type": "error",
                    "message": "SRS file was not generated"
                })
        else:
            queue.append({
                "type": "error",
                "message": "SRS generation failed"
            })

    except Exception as e:
        logger.error(f"Error in SRS generator thread: {str(e)}")
        queue.append({
            "type": "error",
            "message": f"Error during SRS generation: {str(e)}"
        })

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    websocket_id = str(id(websocket))
    
    analyzer = RequirementsAnalyzer()
    analyzer_manager.register_websocket(websocket_id, websocket)
    
    queue_processor = asyncio.create_task(process_analyzer_queue(websocket_id))
    
    try:
        while True:
            data = await websocket.receive_json()
            
            if data["type"] == "analyze":
                thread = threading.Thread(
                    target=run_analyzer_in_thread,
                    args=(analyzer, data["description"], websocket_id)
                )
                thread.start()
            elif data["type"] == "generate_srs":
                thread = threading.Thread(
                    target=run_srs_generator_in_thread,
                    args=(websocket_id,)
                )
                thread.start()
            elif data["type"] == "input_response":
                analyzer_manager.set_input_response(websocket_id, data["value"])

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {websocket_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        analyzer_manager.unregister_websocket(websocket_id)
        queue_processor.cancel()
        try:
            await websocket.close()
        except:
            pass

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)
