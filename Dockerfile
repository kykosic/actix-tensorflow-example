FROM rust:latest AS build

WORKDIR /app
COPY server/Cargo.toml server/Cargo.lock ./
COPY server/src/ src/
RUN cargo build --release
RUN find . -type f -name libtensorflow.so.1 -exec cp {} . \; \
    && find . -type f -name libtensorflow_framework.so.1 -exec cp {} . \;


FROM debian:buster-slim

RUN apt-get update \
    && apt-get install -y \
        ca-certificates \
    && apt-get autoremove \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=build /app/target/release/actix-tf-server /usr/local/bin/actix-tf-server
COPY --from=build /app/*.so.1 /usr/lib/
COPY saved_model/ /app/saved_model/

RUN useradd -mU -s /bin/bash actix
USER actix

EXPOSE 8080
ENTRYPOINT ["actix-tf-server", "--model-dir=/app/saved_model"]
