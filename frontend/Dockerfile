FROM node:20

WORKDIR /frontend

COPY package*.json ./

RUN npm install

COPY . .

EXPOSE 5173
# command when container starts: runds vite dev server
CMD ["npm", "run", "dev", "--", "--host"]