FROM ubuntu:latest
LABEL authors="kunya"

# Install dependencies such as python3, pip3, and git qiskit and matplotlib and numpy
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install qiskit \
    && pip3 install matplotlib \
    && pip3 install numpy

# Copy the current directory contents into the container at /app
COPY . /app

# Set the working directory to
WORKDIR /app

# Run the command to execute the program
CMD ["python3", "main.py"]

# Build the docker image
# docker build -t kunya/qiskit .

# Run the docker image
# docker run -it --rm --name qiskit kunya/qiskit

# Push the docker image to docker hub
# docker push kunya/qiskit

# Pull the docker image from docker hub
# docker pull kunya/qiskit
