FROM python:3.14-slim

# uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# xgboost butuh OpenMP runtime, yang tidak ada di image slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# pakai virtual env hasil uv sync
ENV PATH="/app/.venv/bin:$PATH"

# dependency dulu (layer ini di-cache selama pyproject.toml / uv.lock tidak berubah)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# kode dan model
COPY predict.py model.bin ./

EXPOSE 9696

ENTRYPOINT ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]