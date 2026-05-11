"use client";
import { createContext, useContext, useState, useEffect } from "react";
import { User } from "@/types/user";
import { AuthContextType } from "@/types/auth_context";
import { api, setAccessToken, setOnAuthFailure } from "@/lib/api";
import { PredictionHistoryBase } from "@/types/prediction";
import axios from "axios";

const AuthContext = createContext<AuthContextType | null>(null);

export const AuthProvider = ({ children }: { children: React.ReactNode }) => {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const [newHistoryItem, setNewHistoryItem] = useState<PredictionHistoryBase | null>(null);

  const pushNewHistoryItem = (item: PredictionHistoryBase) => {
    setNewHistoryItem(item);
  };

  useEffect(() => {
    const initializeAuth = async () => {
      try {
        // Gọi refresh để lấy access token mới từ cookie
        const resRefresh = await axios.post("/api/auth/refresh", {}, { withCredentials: true });
        const newAccessToken = resRefresh.data.access_token;

        setAccessToken(newAccessToken);

        const resUser = await api.get("/users/me");
        setUser(resUser.data);
      } catch (error) {
        // Nếu refresh lỗi (hết hạn hoàn toàn), coi như user chưa log in
        setUser(null);
        setAccessToken(null);
      } finally {
        setLoading(false);
      }
    };

    initializeAuth();

    // Register callback to handle auth failures
    const handleAuthFailure = () => {
      setUser(null);
    };
    setOnAuthFailure(handleAuthFailure);

    return () => {
      // Cleanup callback on unmount
      setOnAuthFailure(null);
    };
  }, []);

  const login = (userData: User) => {
    setAccessToken(userData.access_token);
    setUser(userData);
  };

  const updateUser = (userData: Partial<User>) => {
    setUser((prevUser) => (prevUser ? { ...prevUser, ...userData } : null));
  };

  const logout = () => {
    setUser(null);
    setAccessToken(null);
  };

  return (
    <AuthContext.Provider value={{ user, loading, login, logout, newHistoryItem, pushNewHistoryItem, updateUser }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) throw new Error("useAuth must be used within AuthProvider");
  return context;
};