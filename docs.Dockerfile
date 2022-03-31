FROM sphinxdoc/sphinx:4.5.0 AS docs

WORKDIR /app

COPY . .

RUN sphinx-apidoc --full -o ./docs .

RUN cd ./docs && make html

FROM nginx:alpine

COPY --from=docs /app/docs/_build/html/. /usr/share/nginx/html

EXPOSE 80
