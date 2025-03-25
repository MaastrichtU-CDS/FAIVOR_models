fm-build willemsen_tubefeed.py willemsen_tubefeed

# Run model
docker run -d --rm -p 8000:8000 willemsen_tubefeed

curl -X POST -H "Content-type: application/json" -d @input.json http://localhost:8000/predict

curl http://localhost:8000/status

curl http://localhost:8000/result