# Specify the base image for the environment
FROM python:3.8

# Authors of the image

# set a directory for the app
WORKDIR /app

# copy all the files to the container
COPY main.py /app/
COPY requirements.txt /app/
COPY model_execution.py /app/model_execution.py
COPY willemsen_tubefeed.py /app/willemsen_tubefeed.py
COPY modeldata_PEG_tube_feeding.json /app/modeldata_PEG_tube_feeding.json

# install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# specify the port number (from the flask app - 5000)
EXPOSE 8000

# FIXME: pass it as a command to the container in Slicer
CMD ["python", "./main.py"]