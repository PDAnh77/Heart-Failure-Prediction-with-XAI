import { User } from "@/types/user";
import { UnifiedHistoryItem } from "./prediction";

export interface AuthContextType {
  user: User | null;
  loading: boolean;
  newHistoryItem: UnifiedHistoryItem | null;
  login: (userData: User) => void;
  pushNewHistoryItem: (item: UnifiedHistoryItem) => void;
  updateUser: (userData: Partial<User>) => void;
  logout: () => void;
}