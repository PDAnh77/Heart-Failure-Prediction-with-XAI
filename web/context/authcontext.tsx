"use client";
import { createContext, useContext, useState, useEffect } from "react";
import { User } from "@/types/user";
import { AuthContextType } from "@/types/authcontext";
const AuthContext = createContext<AuthContextType | null>(null);
import { api } from "@/lib/api";

export const AuthProvider = ({ children }: { children: React.ReactNode }) => {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const checkUser = async () => {
      try {
        const res = await api.get("/user/me");
        const current_user: User = {
          username: res.data.username,
          email: res.data.email || null
        };
        if (current_user.username) {
          setUser(current_user);
        }
      } catch (error: any) {
        if (error.response?.status === 401) {
          setUser(null);
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
    setUser(userData);
  };

  const logout = () => {
    setUser(null);
  };

  return (
    <AuthContext.Provider value={{ user, loading, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) throw new Error("useAuth must be used within AuthProvider");
  return context;
};