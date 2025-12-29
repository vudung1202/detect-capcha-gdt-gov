from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from pydantic import BaseModel
from .solver import CaptchaSolver
import uvicorn

app = FastAPI()

class SolveRequest(BaseModel):
    svg_content: str

solver = CaptchaSolver()

@app.on_event("startup")
def load_solver():
    solver.load_db()
    print(f"Solver loaded with {len(solver.knowledge_base)} patterns.")

@app.post("/solve")
async def solve_captcha(
    request: Request,
    file: UploadFile = File(None),
    url: str = Form(None)
):
    try:
        svg = ""
        
        # 1. Handle File Upload
        if file:
            content = await file.read()
            svg = content.decode('utf-8')
            
        # 2. Handle URL (Form data or JSON)
        elif url:
            import requests
            # Use sync request for simplicity in this mini-project
            resp = requests.get(url, timeout=10, verify=False)
            resp.raise_for_status()
            content_type = resp.headers.get('Content-Type', '')
            if 'application/json' in content_type:
                data = resp.json()
                svg = data.get('content') or data.get('svg')
            elif 'image/png' in content_type or resp.content.startswith(b'\x89PNG'):
                 svg = resp.content # Treat as bytes
            else:
                svg = resp.text
                
        # 3. Handle Raw/JSON Body
        else:
            content_type = request.headers.get('content-type', '')
            if 'application/json' in content_type:
                try:
                    data = await request.json()
                    # Check if 'url' provided in JSON body
                    if 'url' in data:
                        import requests
                        resp = requests.get(data['url'], timeout=10, verify=False)
                        resp.raise_for_status()
                        fetched_type = resp.headers.get('Content-Type', '')
                        if 'application/json' in fetched_type:
                             j = resp.json()
                             svg = j.get('content') or j.get('svg')
                        elif 'image/png' in fetched_type or resp.content.startswith(b'\x89PNG'):
                             svg = resp.content # Treat as bytes
                        else:
                             svg = resp.text
                    else:
                        svg = data.get('content') or data.get('svg')
                except Exception:
                    pass
            
            # If still empty, try raw body
            if not svg:
                body = await request.body()
                if body:
                    # Check if it's a PNG byte stream
                    if body.startswith(b'\x89PNG'):
                        svg = body
                    else:
                        try:
                            svg = body.decode('utf-8')
                        except UnicodeDecodeError:
                            # Might be binary but not PNG? default directly
                            svg = body

        if not svg:
            raise HTTPException(status_code=400, detail="No CAPTCHA content provided (file, url, or body)")
        
        # Dispatch based on type
        # If bytes and PNG header -> PNG
        from .utils import process_png_content
        
        if isinstance(svg, bytes) and svg.startswith(b'\x89PNG'):
            paths = process_png_content(svg)
            text = solver.solve(paths)
        else:
            # Assume SVG string
            text = solver.solve(svg)
            
        return {"text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def health():
    return {"status": "ok", "patterns_loaded": len(solver.knowledge_base)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
