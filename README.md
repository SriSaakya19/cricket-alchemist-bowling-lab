# The Cricket Alchemist — VS Code project

## Files
- `app.py` — Streamlit UI and charts
- `dynamic_processor.py` — sampled video processing, pose overlays, approximate 2D angle indicators
- `requirements.txt` — Python dependencies

## Recommended setup (Windows)
Use **Python 3.11, 64-bit** for compatibility with the pinned MediaPipe package.

Open the VS Code terminal in this project folder and run:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation, run this instead in the same terminal:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

The app usually opens at `http://localhost:8501`.

## Notes
The pose overlay is an estimated 2D landmark view. Joint angles are approximate image-plane indicators and not validated biomechanical measurements. The app does not determine ball speed, injury risk, 3D biomechanics or bowling legality. Test with short, well-lit videos before using it in a real coaching workflow.
