import ky from 'ky';

export const api = ky.create({
  prefixUrl: '/api',
  headers: { 'Content-Type': 'application/json' },
  timeout: 60_000,
  retry: {
    limit: 2,
    methods: ['get'],
    statusCodes: [408, 429, 500, 502, 503],
  },
});