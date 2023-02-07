# docker run --rm --name nginx_server -p 8080:8080 -v /tmp/mlreports:/tmp/mlreports nginx_server

FROM nginx:alpine

WORKDIR /app

COPY ./nginx.conf /etc/nginx/nginx.conf

EXPOSE 8888

ENTRYPOINT ["nginx", "-g", "daemon off;"]
