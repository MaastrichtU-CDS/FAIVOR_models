# local build
# fm-build willemsen_tubefeed.py willemsen_tubefeed

# Run model
docker run -d --rm --name tubefeed -p 8000:8000 ghcr.io/maastrichtu-cds/faivor_models/willemsen_tubefeed:latest

wait 20

curl -X POST -H "Content-type: application/json" -d @input.json http://localhost:8000/predict

wait 5

curl http://localhost:8000/status

wait 5

curl http://localhost:8000/result

docker stop tubefeed