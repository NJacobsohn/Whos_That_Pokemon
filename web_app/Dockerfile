# Use an official Python runtime as a parent image
FROM python:latest

# Set the working directory to /app
WORKDIR /web_app

# Copy the current directory contents into the container at /app
COPY . /web_app

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 8080


# Run app.py when the container launches
CMD ["python", "app.py"]
