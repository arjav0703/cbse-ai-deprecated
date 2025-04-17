FROM node:20-alpine

WORKDIR /app

COPY package*.json ./

RUN npm install
RUN npm install pm2@latest

COPY . .

EXPOSE 8080 8081 8082

# CMD ["node", "server.js"]
CMD ["pm2-runtime", "start", "app1/server.js", "--name", "app1", "--watch", "&&", "pm2-runtime", "start", "sci-server.js", "--name", "Science", "--watch", "&&", "pm2-runtime", "start", "eng-server.js", "--name", "English", "--watch"]
