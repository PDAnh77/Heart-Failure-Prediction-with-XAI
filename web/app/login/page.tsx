"use client"
import { useRouter } from "next/navigation";
import { useState, useEffect } from "react"
import { toast } from 'react-toastify';
import { useAuth } from "@/context/authcontext";
import { User } from "@/types/user";
import { FcGoogle } from "react-icons/fc";
import { api } from "@/lib/api";

export default function Login() {
    const [username, setUsername] = useState("")
    const [password, setPassword] = useState("")
    const [loading, setLoading] = useState(false)
    const router = useRouter()
    const { login, user } = useAuth();

    useEffect(() => {
        if (user) {
            router.push("/predict")
        }
    }, [user])

    const handleGoogleLogin = () => {
        window.location.href = "/api/auth/google";
    }

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault()
        setLoading(true)
        toast.dismiss();

        const payload = {
            username: username,
            password: password
        }

        try {
            const res = await api.post("/auth/login", payload)
            const current_user: User = res.data
            login({ username: current_user.username, email: null })
            router.push("/predict")
        } catch (error: any) {
            if (error.response?.status === 401) {
                toast.error("Invalid username or password.");
                return;
            }
            toast.error("An error occurred. Please try again later.");
        } finally {
            setLoading(false)
        }
    }

    return (
        <div className="flex min-h-full flex-col px-6 py-12 lg:px-8">
            <div className="sm:mx-auto sm:w-full sm:max-w-sm">
                <h2 className="mt-10 text-center text-2xl/9 font-bold tracking-tight text-gray-900 dark:text-white">
                    Sign in to your account
                </h2>
            </div>

            <div className="mt-10 sm:mx-auto sm:w-full sm:max-w-sm">
                <form onSubmit={handleSubmit} method="POST" className="space-y-6">
                    <div>
                        <label htmlFor="username" className="block text-md/6 font-medium text-gray-900 dark:text-gray-100">
                            Username
                        </label>
                        <div className="mt-2">
                            <input
                                id="username"
                                name="username"
                                type="text"
                                value={username || ""}
                                onChange={(e) => setUsername(e.target.value)}
                                required
                                placeholder="Username"
                                autoComplete="username"
                                className="block w-full rounded-xl bg-white shadow-sm transition px-3 py-2 text-base text-gray-900 outline-1 -outline-offset-1 outline-gray-300 placeholder:text-gray-400 focus:outline-2 focus:-outline-offset-2 focus:outline-indigo-600 sm:text-base/6 dark:bg-white/5 dark:text-white dark:outline-white/10 dark:placeholder:text-gray-500 dark:focus:outline-indigo-500"
                            />
                        </div>
                    </div>

                    <div>
                        <div className="flex items-center justify-between">
                            <label htmlFor="password" className="block text-md/6 font-medium text-gray-900 dark:text-gray-100">
                                Password
                            </label>
                            <div className="text-sm">
                                <a
                                    href="#"
                                    tabIndex={-1}
                                    className="hidden font-semibold text-indigo-600 hover:text-indigo-500 dark:text-indigo-400 dark:hover:text-indigo-300"
                                >
                                    Forgot password?
                                </a>
                            </div>
                        </div>
                        <div className="mt-2">
                            <input
                                id="password"
                                name="password"
                                type="password"
                                value={password || ""}
                                onChange={(e) => setPassword(e.target.value)}
                                required
                                placeholder="Password"
                                autoComplete="current-password"
                                className="block w-full rounded-xl shadow-sm transition bg-white px-4 py-2 text-base text-gray-900 outline-1 -outline-offset-1 outline-gray-300 placeholder:text-gray-400 focus:outline-2 focus:-outline-offset-2 focus:outline-indigo-600 sm:text-base/6 dark:bg-white/5 dark:text-white dark:outline-white/10 dark:placeholder:text-gray-500 dark:focus:outline-indigo-500"
                            />
                        </div>
                    </div>

                    <div>
                        <button
                            type="submit"
                            disabled={loading}
                            className="flex w-full transition justify-center rounded-md bg-indigo-600 px-3 py-2 text-base/6 font-semibold text-white shadow-xs hover:bg-indigo-500 cursor-pointer focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-600 dark:bg-indigo-500 dark:shadow-none dark:hover:bg-indigo-400 dark:focus-visible:outline-indigo-500"
                        >
                            {loading ? "Signing in..." : "Sign in"}
                        </button>
                    </div>
                </form>

                <div className="mt-6">
                    <div className="relative">
                        <div className="absolute inset-0 flex items-center">
                            <span className="w-full border-t border-gray-300 dark:border-gray-600" />
                        </div>
                        <div className="relative flex justify-center text-sm">
                            <span className="bg-white dark:bg-[#0a0a0a] px-2 text-gray-500 dark:text-gray-400">
                                Or sign in with
                            </span>
                        </div>
                    </div>

                    <div className="mt-6">
                        <button
                            onClick={handleGoogleLogin}
                            type="button"
                            className="flex w-full transition items-center justify-center gap-2 rounded-md bg-white px-3 py-2 border border-gray-200 dark:border-gray-600 text-base/6 font-semibold text-gray-900 shadow-sm hover:bg-gray-50 cursor-pointer dark:bg-white/10 dark:text-white dark:hover:bg-white/20"
                        >
                            <FcGoogle className="text-2xl" />
                            Google
                        </button>
                    </div>
                </div>
            </div>
        </div>
    )
}