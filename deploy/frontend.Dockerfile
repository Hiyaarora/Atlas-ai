# ==========================================================================
# Atlas AI frontend image
#
# `development` runs the Vite dev server with HMR.
# `production` builds static assets and serves them via nginx.
# ==========================================================================

FROM node:22-alpine AS base
WORKDIR /app

# ---- development ---------------------------------------------------------
FROM base AS development

# Copy manifests first so `npm ci` is cached until dependencies actually change.
COPY package*.json ./
RUN npm install

COPY . .

EXPOSE 5173
CMD ["npm", "run", "dev"]

# ---- build ---------------------------------------------------------------
FROM base AS build

COPY package*.json ./
RUN npm ci

COPY . .
RUN npm run build

# ---- production ----------------------------------------------------------
FROM nginx:1.27-alpine AS production

COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
