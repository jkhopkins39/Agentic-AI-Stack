const DEFAULT_API_URL = 'http://localhost:8000';

const normalizedApiUrl = (import.meta.env.VITE_API_URL || DEFAULT_API_URL).replace(/\/$/, '');

const toWebSocketUrl = (url: string) => {
  if (url.startsWith('https://')) {
    return `wss://${url.slice('https://'.length)}`;
  }
  if (url.startsWith('http://')) {
    return `ws://${url.slice('http://'.length)}`;
  }
  return url.replace(/^http/, 'ws');
};

export const API_BASE_URL = normalizedApiUrl;
export const WS_BASE_URL = toWebSocketUrl(normalizedApiUrl);

