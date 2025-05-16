# Use a Windows Server base image
FROM mcr.microsoft.com/windows/servercore:19045

# Set environment variables
ENV PYTHON_VERSION=3.9.8

# Install Python
RUN curl -o python.exe https://www.python.org/ftp/python/%PYTHON_VERSION%/python-%PYTHON_VERSION%-amd64.exe && \
    python.exe /quiet InstallAllUsers=1 PrependPath=1 Include_test=0 && \
    del python.exe

# Copy in the entire project (including PrePoMax, scripts, reference files, etc.)
COPY . C:/determineMaterialProperties

# Set working directory
WORKDIR C:/determineMaterialProperties

# Install pip packages
COPY requirements.txt .
RUN pip install -r requirements.txt

# Run script
ENTRYPOINT ["python", "runOptimization.py"]
