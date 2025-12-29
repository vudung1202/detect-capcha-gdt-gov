from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from pydantic import BaseModel
from .solver import CaptchaSolver
import uvicorn
import requests
import asyncio
from concurrent.futures import ThreadPoolExecutor

app = FastAPI()

class SolveRequest(BaseModel):
    svg_content: str = None
    url: str = None

solver = CaptchaSolver()

# Create a thread pool for blocking I/O
executor = ThreadPoolExecutor(max_workers=10)

@app.on_event("startup")
def load_solver():
    solver.load_db()
    print(f"Solver loaded with {len(solver.knowledge_base)} patterns.")

def fetch_url_content(url: str):
    """
    Synchronous helper to fetch content from URL.
    Returns: (content_bytes, is_png_bool)
    """
    try:
        resp = requests.get(url, timeout=10, verify=False)
        resp.raise_for_status()
        
        content_type = resp.headers.get('Content-Type', '').lower()
        is_png = 'image/png' in content_type or resp.content.startswith(b'\x89PNG')
        
        # If JSON response (some APIs wrap the SVG), extract it
        if 'application/json' in content_type:
            try:
                data = resp.json()
                content = data.get('content') or data.get('svg')
                if not content:
                    return None, False
                if isinstance(content, str):
                    return content.encode('utf-8'), False
                return content, False # generic bytes
            except:
                return resp.content, is_png
                
        return resp.content, is_png
    except Exception as e:
        print(f"Error fetching URL {url}: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to fetch URL: {str(e)}")

@app.post("/solve")
async def solve_captcha(
    request: Request,
    file: UploadFile = File(None),
    url: str = Form(None)
):
    try:
        content_bytes = None
        is_png = False

        # 1. Handle File Upload
        if file:
            content_bytes = await file.read()
            if content_bytes.startswith(b'\x89PNG'):
                is_png = True
            
        # 2. Handle URL (Form data)
        elif url:
            loop = asyncio.get_event_loop()
            content_bytes, is_png = await loop.run_in_executor(executor, fetch_url_content, url)

        # 3. Handle JSON Body / Raw Body
        else:
            content_type = request.headers.get('content-type', '').lower()
            if 'application/json' in content_type:
                try:
                    data = await request.json()
                    # Check for 'url' in JSON body
                    if 'url' in data and data['url']:
                        loop = asyncio.get_event_loop()
                        content_bytes, is_png = await loop.run_in_executor(executor, fetch_url_content, data['url'])
                    else:
                        svg_str = data.get('content') or data.get('svg') or data.get('svg_content')
                        if svg_str:
                            content_bytes = svg_str.encode('utf-8')
                except Exception:
                    pass
            
            # If still nothing, try raw body
            if not content_bytes:
                body = await request.body()
                if body:
                    content_bytes = body
                    if body.startswith(b'\x89PNG'):
                        is_png = True

        if not content_bytes:
            raise HTTPException(status_code=400, detail="No CAPTCHA content provided (file, url, or body)")
        
        # Dispatch to solver
        if is_png:
            from .utils import process_png_content
            # process_png_content expects bytes
            paths = process_png_content(content_bytes)
            text = solver.solve(paths)
        else:
            # Assume SVG string
            try:
                svg_text = content_bytes.decode('utf-8')
                text = solver.solve(svg_text)
            except UnicodeDecodeError:
                # Fallback if it was binary but not PNG? or invalid encoding
                # Attempt to solve anyway or fail
                 raise HTTPException(status_code=400, detail="Invalid SVG encoding")
            
        return {"text": text}
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def health():
    return {"status": "ok", "patterns_loaded": len(solver.knowledge_base)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
