FROM tensorflow/tensorflow:2.12.0-gpu
WORKDIR /workspace

RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

RUN python -m pip uninstall -y jax jaxlib || true


COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["bash"]