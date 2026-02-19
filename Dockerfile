FROM python:3.12-bookworm

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install Azure CLI
RUN curl -sL https://aka.ms/InstallAzureCLIDeb | bash

WORKDIR /workspace

# Cache dependencies — only re-installed when lock/pyproject change
COPY pyproject.toml uv.lock ./
RUN uv sync --no-install-project --frozen

# Copy source (invalidated more often than deps)
COPY src/ src/
COPY tests/ tests/

CMD ["/bin/bash"]
