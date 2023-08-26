# start from nvidia tensorflow
FROM nvcr.io/nvidia/pytorch:21.04-py3

# install packages
COPY requirements.txt ./

RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt
