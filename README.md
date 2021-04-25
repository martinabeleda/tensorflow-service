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

To run a local development instance with `docker compose`:

```bash
docker compose up
```

The service is now available at http://localhost:8080
