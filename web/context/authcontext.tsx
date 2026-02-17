"use client";
import { createContext, useContext, useState, useEffect } from "react";
import { User } from "@/types/user";
import { AuthContextType } from "@/types/authcontext";
import { api, setAccessToken } from "@/lib/api";
import { PredictionHistoryBase } from "@/types/prediction";

const AuthContext = createContext<AuthContextType | null>(null);

export const AuthProvider = ({ children }: { children: React.ReactNode }) => {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const [newHistoryItem, setNewHistoryItem] = useState<PredictionHistoryBase | null>(null);

  const pushNewHistoryItem = (item: PredictionHistoryBase) => {
    setNewHistoryItem(item);
  };

  useEffect(() => {
    const checkUser = async () => {
      try {
        const res = await api.get("/user/me");
        setUser(res.data);
      } catch (error: any) {
        if (error.response?.status === 401) {
          setUser(null);
          setAccessToken(null);
        } else {
          console.error("Auth check failed:", error);
        }
      } finally {
        setLoading(false);
      }
    };
    checkUser();
  }, []);

  const login = (userData: User) => {
    setAccessToken(userData.access_token);
    setUser(userData);
  };

  const logout = () => {
    setUser(null);
    setAccessToken(null);
  };

  return (
    <AuthContext.Provider value={{ user, loading, login, logout, newHistoryItem, pushNewHistoryItem }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) throw new Error("useAuth must be used within AuthProvider");
  return context;
};