# Installation


## Using Python Environements
You can run this application in a dedicated python environement.
```
git clone https://github.com/rayenebech/finance-assistance.git
cd finance-assistance
python3 -m venv mvenv
source mvenv/bin/activate
pip install -U pip
pip install -r requirements.txt
```
After that the application can be launched by running 
````
streamlit run app.py --server.enableCORS false --server.enableXsrfProtection false
````
## Using Docker
Otherwise, You can also run this application using 
```
git clone https://github.com/rayenebech/finance-assistance.git
cd finance-assistance
docker-compose up --build -d
```
The docker container finance-assistant will launch a `Streamlit` application. You can view the logs of the docker container by running:
```
docker logs -f --tail 500 finance-assistant
```
