FROM python:3.12-slim-bookworm

WORKDIR /app

# Copy the requirements file
COPY ./requirements.txt .

# Install Python deps
RUN pip install --no-cache-dir -r requirements.txt

COPY ./src .

# Expose the port uvicorn will listen on
EXPOSE 4000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "4000"]