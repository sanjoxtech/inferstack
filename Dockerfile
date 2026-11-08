# InferStack control plane / gateway / worker image (CPU). GPU engines run in their own images.
# Base pinned by digest; dependencies installed from the hashed lock (pip --require-hashes).
FROM python:3.12-slim@sha256:a0f57064cbdcd6285fa45f51237ab0b35ea8ebaed06a1f5973e89e19dfb86704 AS build
WORKDIR /src
COPY requirements.lock ./
RUN python -m venv /opt/venv && /opt/venv/bin/pip install --no-cache-dir --require-hashes -r requirements.lock
COPY pyproject.toml README.md ./
COPY src ./src
RUN /opt/venv/bin/pip install --no-cache-dir --no-deps .

FROM python:3.12-slim@sha256:a0f57064cbdcd6285fa45f51237ab0b35ea8ebaed06a1f5973e89e19dfb86704
RUN apt-get update && apt-get install -y --no-install-recommends bubblewrap git \
    && rm -rf /var/lib/apt/lists/* && useradd --uid 65532 --create-home inferstack
COPY --from=build /opt/venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
USER 65532
WORKDIR /home/inferstack
ENTRYPOINT ["python", "-m", "inferstack.control"]
CMD ["serve", "--host", "0.0.0.0", "--state", "/home/inferstack/state"]
