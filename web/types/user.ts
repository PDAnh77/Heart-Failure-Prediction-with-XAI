export interface User {
  username: string;
  access_token: string;
  email?: string | null;
  avatar_url?: string | null;
  display_name?: string | null;
}