"use client"

import { useState } from "react"
import { useRouter } from "next/navigation"
import { toast } from 'react-toastify';

export default function Login() {
    const router = useRouter()
    const [username, setUsername] = useState("")
    const [password, setPassword] = useState("")
    const [loading, setLoading] = useState(false)

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault()
        setLoading(true)
        toast.dismiss(); // Xóa các toast cũ trước khi submit mới

        try {
            const body = new URLSearchParams()
            body.append("username", username)
            body.append("password", password)

            const res = await fetch("https://heart-failure-api-uwqj.onrender.com/api/user/auth", {
                method: "POST",
                headers: {
                    "Content-Type": "application/x-www-form-urlencoded",
                },
                body,
            })

            if (!res.ok) {
                const errorData = await res.json().catch(() => null)
                throw new Error(errorData?.detail || "Invalid username or password.")
            }

            const data = await res.json()

            localStorage.setItem("access_token", data.access_token)
            localStorage.setItem("token_type", data.token_type)
            localStorage.setItem("username", username)

            toast.success("Login successful.", { autoClose: 1500 });
            setTimeout(() => {
                window.location.href = "/predict"
            }, 2000)

        } catch (err: any) {
            if (err.response?.status === 401) {
                toast.error("Invalid username or password.");
            } else {
                toast.error("An error occurred. Please try again later.");
            }
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
                                className="block w-full rounded-xl bg-white px-3 py-2 text-base text-gray-900 outline-1 -outline-offset-1 outline-gray-300 placeholder:text-gray-400 focus:outline-2 focus:-outline-offset-2 focus:outline-indigo-600 sm:text-base/6 dark:bg-white/5 dark:text-white dark:outline-white/10 dark:placeholder:text-gray-500 dark:focus:outline-indigo-500"
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
                                    className="font-semibold text-indigo-600 hover:text-indigo-500 dark:text-indigo-400 dark:hover:text-indigo-300"
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
                                className="block w-full rounded-xl bg-white px-4 py-2 text-base text-gray-900 outline-1 -outline-offset-1 outline-gray-300 placeholder:text-gray-400 focus:outline-2 focus:-outline-offset-2 focus:outline-indigo-600 sm:text-base/6 dark:bg-white/5 dark:text-white dark:outline-white/10 dark:placeholder:text-gray-500 dark:focus:outline-indigo-500"
                            />
                        </div>
                    </div>

                    <div>
                        <button
                            type="submit"
                            disabled={loading}
                            className="flex w-full justify-center rounded-md bg-indigo-600 px-3 py-1.5 text-base/6 font-semibold text-white shadow-xs hover:bg-indigo-500 cursor-pointer focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-600 dark:bg-indigo-500 dark:shadow-none dark:hover:bg-indigo-400 dark:focus-visible:outline-indigo-500"
                        >
                            {loading ? "Signing in..." : "Sign in"}
                        </button>
                    </div>
                </form>
            </div>
        </div>
    )
}