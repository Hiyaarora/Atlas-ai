# ==========================================================================
# Atlas AI frontend image
#
# `development` runs the Vite dev server with HMR.
# `production` builds static assets and serves them through nginx, which also
# proxies /api to the backend so the browser talks to a single origin.
# ==========================================================================

FROM node:22-alpine AS base
WORKDIR /app

# ---- development ---------------------------------------------------------
FROM base AS development

COPY package*.json ./
RUN npm ci
COPY . .

EXPOSE 5173
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]

# ---- build ---------------------------------------------------------------
FROM base AS build

COPY package*.json ./
# `npm ci` not `npm install`: it installs exactly the lockfile, so an image
# built today and one built next month contain the same dependency tree.
RUN npm ci

COPY . .

# Vite inlines VITE_* at BUILD time, so this cannot be supplied at run time.
#
# "/" resolves to an empty base in config/env.ts, which makes every request a
# same-origin relative path. That is the point: nginx proxies /api to the
# backend, so there is no cross-origin request, no CORS preflight, and the
# httpOnly refresh cookie is a first-party cookie rather than one that needs
# SameSite=None to survive.
ARG VITE_API_BASE_URL=/
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL

RUN npm run build

# ---- production ----------------------------------------------------------
FROM nginx:1.27-alpine AS production

COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
