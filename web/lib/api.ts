const BASE_URL = process.env.NEXT_PUBLIC_API_URL;

export async function apiFetch(path: string, options: RequestInit = {}) {
  return fetch(`${BASE_URL}/api${path}`, {
    headers: {
      "Content-Type": "application/json"
    },
    ...options,
  });
}
