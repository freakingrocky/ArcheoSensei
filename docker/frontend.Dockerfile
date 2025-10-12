FROM node:20-alpine
WORKDIR /app
COPY frontend/package.json frontend/pnpm-lock.yaml* frontend/ /app/
RUN npm i -g pnpm
COPY frontend /app
ENV CI=true
EXPOSE 3000
CMD ["sh", "-c", "pnpm install && pnpm dev -p 3000"]
