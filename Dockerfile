# DistillGapBench CPU image: runs the test suite and the sandbox check without GPUs or network.
#   docker build -t distillgapbench:cpu .
#   docker run --rm --network none --security-opt seccomp=unconfined --security-opt apparmor=unconfined distillgapbench:cpu
FROM docker.io/library/debian:bookworm-slim

# util-linux unshare: the namespace sandbox for researcher-written code (the mechanism used in the study; bubblewrap
# is deliberately not installed); procps: the memory watchdog (ps)
RUN apt-get update \
 && apt-get install -y --no-install-recommends util-linux procps ca-certificates \
 && rm -rf /var/lib/apt/lists/*

COPY --from=docker.io/astral/uv:0.13.6 /uv /usr/local/bin/uv

# The same standalone CPython as the study (3.12.13). It also runs the sandboxed code, so it lives under /opt, which
# the sandbox hides; the sandbox stages a private copy of the interpreter.
ENV UV_PYTHON_INSTALL_DIR=/opt/python UV_NO_CACHE=1 UV_PYTHON_PREFERENCE=only-managed
RUN uv python install 3.12.13

WORKDIR /opt/distillgapbench
COPY requirements.txt .
RUN uv venv --python 3.12.13 /opt/distillgapbench/.venv \
 && uv pip install --python /opt/distillgapbench/.venv/bin/python -r requirements.txt
COPY . .
# readable by any user (Apptainer runs as the host user; a strict umask on the host must not leak into the image)
RUN chmod -R a+rX /opt/distillgapbench /opt/python

# Code, keys and results stay under paths the sandbox hides (/opt, /home); run as an unprivileged user.
RUN useradd --create-home --uid 1000 dg && mkdir -p /home/dg/runs && chown -R dg:dg /home/dg
USER dg
ENV HOME=/home/dg PATH=/opt/distillgapbench/.venv/bin:$PATH PYTHONDONTWRITEBYTECODE=1
CMD ["bash", "scripts/selfcheck.sh"]
