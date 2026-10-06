.PHONY: install dev run mock-ehr test eval lint docker
install:   ; pip install -e ".[dev]"
run:       ; aurelia serve
mock-ehr:  ; aurelia mock-ehr
test:      ; pytest -q
eval:      ; python -m eval.benchmark
lint:      ; ruff check .
docker:    ; docker compose up --build
