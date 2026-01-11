import { User } from "@/types/user";

export interface AuthContextType {
  user: User | null;
  loading: boolean;
  refreshHistoryTicket: number | null;
  login: (userData: User) => void;
  triggerRefreshHistory: () => void;
  logout: () => void;
}