FROM python:3.9-slim

WORKDIR /app

# copy the contents from local DIR . / to app DIR  
COPY . /app 

# install req while removing previous cache
RUN pip install --upgrade pip
RUN pip install --no-cache-dir --default-timeout=100 tensorflow==2.19.0
RUN pip install --no-cache-dir -e .

# Expose the flask port
EXPOSE 5000

CMD ["python","application.py"]