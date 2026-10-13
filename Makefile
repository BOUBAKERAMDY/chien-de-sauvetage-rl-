.PHONY: install test view clean

install:            ## Environnement uv (.venv, Python 3.12) + dépendances
	uv sync --extra dev

test:               ## Tests unitaires
	uv run pytest -q

view:               ## Le chien dans le viewer 3D (Mac : mjpython obligatoire)
	uv run mjpython scripts/view_dog.py

clean:
	find . -path ./.venv -prune -o -name __pycache__ -type d -exec rm -rf {} +
