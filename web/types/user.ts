export interface User {
  username: string;
  access_token: string;
  email: string | null;
  avatar_url?: string | null;
}