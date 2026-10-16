# The container the experimental coding agent (core/agent_coder.py, ClaudeCliRunner) runs in; built once on first use.
# Claude Code runs here with --permission-mode bypassPermissions because this container IS the sandbox: only the quest's
# code/ folder and a per-quest folder for the agent's session files are mounted into it. No login is baked in: the token
# comes in as an environment variable passed by name (CLAUDE_CODE_OAUTH_TOKEN) when the container starts.
FROM python:3.11-slim-bookworm

ARG CLAUDE_VERSION=2.1.299

RUN apt-get update \
    && apt-get install -y --no-install-recommends nodejs npm ca-certificates git \
    && rm -rf /var/lib/apt/lists/* \
    && npm install -g @anthropic-ai/claude-code@${CLAUDE_VERSION} \
    && npm cache clean --force

# What a study usually needs; the quest's own code/requirements.txt is installed when the container starts.
ARG SCIENCE_PACKAGES="numpy scipy matplotlib pandas sympy pytest"
RUN pip install --no-cache-dir --retries 10 --timeout 30 ${SCIENCE_PACKAGES}

# A user that is not root (the CLI refuses to bypass permissions as root) with a passwd entry (Node's os.userInfo()
# needs one); the folders are writable by any uid so the host user's id can be used with --user.
RUN useradd -u 1000 -m -d /home/agent agent \
    && mkdir -p /home/agent/.claude /work/code && chmod 777 /home/agent /home/agent/.claude /work /work/code
ENV HOME=/home/agent PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg DISABLE_AUTOUPDATER=1 \
    CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 DISABLE_TELEMETRY=1 DISABLE_ERROR_REPORTING=1
USER agent
WORKDIR /work/code
