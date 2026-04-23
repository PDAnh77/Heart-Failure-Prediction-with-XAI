import { User } from "@/types/user";
import { PredictionHistoryBase } from "./prediction";

export interface AuthContextType {
  user: User | null;
  loading: boolean;
  newHistoryItem: PredictionHistoryBase | null;
  login: (userData: User) => void;
  pushNewHistoryItem: (item: PredictionHistoryBase) => void;
  updateUser: (userData: Partial<User>) => void;
  logout: () => void;
}