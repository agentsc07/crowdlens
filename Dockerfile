# Hugging Face Space (Docker SDK, free CPU). Also runs anywhere: docker build -t crowdlens . && docker run -p 7860:7860 crowdlens
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH HF_HOME=/home/user/.cache/huggingface \
    CROWDLENS_LITE=1 CROWDLENS_DATA=/home/user/app/space_data CROWDLENS_DEVICE=cpu PYTHONUNBUFFERED=1
WORKDIR /home/user/app
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY --chown=user pyproject.toml README.md ./
COPY --chown=user crowdlens ./crowdlens
RUN pip install --no-cache-dir .
# bake the models in so the first visitor doesn't wait for downloads
RUN python -c "from huggingface_hub import snapshot_download as d; d('Qwen/Qwen3-VL-2B-Instruct'); d('google/siglip2-base-patch16-224')" \
 && python -c "from rfdetr import RFDETRNano; RFDETRNano()"
COPY --chown=user .streamlit ./.streamlit
COPY --chown=user space_data ./space_data
EXPOSE 7860
CMD ["streamlit", "run", "crowdlens/app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true", \
     "--server.enableXsrfProtection=false", "--server.enableCORS=false", "--server.maxUploadSize=60"]
