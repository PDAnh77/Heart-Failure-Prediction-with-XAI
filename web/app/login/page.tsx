"use client"
import { useState } from "react"
import { toast } from 'react-hot-toast';
import { useAuth } from "@/context/authcontext";
import { User } from "@/types/user";
import { FcGoogle } from "react-icons/fc";
import { api } from "@/lib/api";

export default function Login() {
    const [loginId, setLoginId] = useState("")
    const [password, setPassword] = useState("")
    const [errors, setErrors] = useState({ loginId: "", password: "" })
    const [loading, setLoading] = useState(false)
    const { login } = useAuth();

    const handleGoogleLogin = () => {
        window.location.href = "/api/auth/google";
    }

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault()
        setErrors({ loginId: "", password: "" })
        toast.dismiss();
        let hasError = false;
        const newErrors = { loginId: "", password: "" };

        if (!loginId.trim()) {
            newErrors.loginId = "Please enter username or email.";
            hasError = true;
        }
        if (!password) {
            newErrors.password = "Please enter password.";
            hasError = true;
        }

        if (hasError) {
            setErrors(newErrors);
            return;
        }
        setLoading(true)

        const payload = {
            login_id: loginId,
            password: password
        }

        try {
            const res = await api.post("/auth/login", payload)
            const current_user: User = res.data
            login(current_user)
            setLoginId("")
            setPassword("")
            toast.success("Signed in successfully.")
        } catch (error: any) {
            if (error.response?.status === 401) {
                toast.error("Invalid login info");
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
                <form onSubmit={handleSubmit} method="POST" noValidate className="space-y-6">
                    <div>
                        <label htmlFor="loginId" className="block text-md/6 font-medium text-gray-900 dark:text-gray-100">
                            Username or Email
                        </label>
                        <div className="mt-2">
                            <input
                                id="loginId"
                                name="login_id"
                                type="text"
                                value={loginId || ""}
                                onChange={(e) => {
                                    setLoginId(e.target.value);
                                    if (errors.loginId) {
                                        setErrors((prev) => ({ ...prev, loginId: "" }));
                                    }
                                }}
                                required
                                placeholder="Username or Email"
                                className={`block w-full rounded-xl shadow-sm transition bg-white px-4 py-2 text-base text-gray-900 outline-1 -outline-offset-1 placeholder:text-gray-400 focus:outline-2 focus:-outline-offset-2 sm:text-base/6 dark:bg-white/5 dark:text-white dark:placeholder:text-gray-500
                                    ${errors.loginId
                                        ? "outline-red-500 focus:outline-red-500"
                                        : "outline-gray-300 focus:outline-indigo-600 dark:outline-white/10 dark:focus:outline-indigo-500"
                                    }`}
                            />
                            {errors.loginId && (
                                <p className="mt-1 text-sm text-red-500 animate-in fade-in duration-200">
                                    {errors.loginId}
                                </p>
                            )}
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
                                onChange={(e) => {
                                    setPassword(e.target.value);
                                    if (errors.password) {
                                        setErrors((prev) => ({ ...prev, password: "" }));
                                    }
                                }}
                                required
                                placeholder="Password"
                                autoComplete="current-password"
                                className={`block w-full rounded-xl shadow-sm transition bg-white px-4 py-2 text-base text-gray-900 outline-1 -outline-offset-1 placeholder:text-gray-400 focus:outline-2 focus:-outline-offset-2 sm:text-base/6 dark:bg-white/5 dark:text-white dark:placeholder:text-gray-500
                                    ${errors.password
                                        ? "outline-red-500 focus:outline-red-500"
                                        : "outline-gray-300 focus:outline-indigo-600 dark:outline-white/10 dark:focus:outline-indigo-500"
                                    }`}
                            />
                            {errors.password && (
                                <p className="mt-1 text-sm text-red-500 animate-in fade-in duration-200">
                                    {errors.password}
                                </p>
                            )}
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