# AI Automation Engineer Challenge

A service for serving model predictions. See the [documentation](./DOCS.md) for a discussion on any design decisions.

## Development

See the [official documentation](https://python-poetry.org/docs/#installation) for installing poetry.

To set up a local dev environment, create a `virtualenv` using poetry:

```bash
poetry shell
```

Install dependencies:

```bash
poetry install
```

Build the container if it doesn't exist or you've updated any dependencies:

```bash
docker compose build
```

To run a local development instance with `docker compose`. Note that this mounts the local dev directory in the
container and uses `uvicorn` direcly in `--reload` mode so that changes are detected and reloaded automatically:

```bash
docker compose up
```

The service is now available at http://localhost:8080/docs

`pre-commit` has been configured to run autostyling, linting and tests on commit. To initialize, run:

```bash
pre-commit install
```

## Load test

First, deploy the dockerised solution with the default `CMD` which uses `gunicorn` to manage `uvicorn` and this will
do all of the heavy lifting w.r.t. managing multiple workers:

```bash
# Create a tagged build
export VERSION=0.1.0
docker build -f Dockerfile -t martinabeleda/ai-auto-challenge:${VERSION} .

docker run -d -p 80:80 --name load-test martinabeleda/ai-auto-challenge:${VERSION}
```

Run the load test using locust:

```bash
locust -f tests/locustfile.py
```
