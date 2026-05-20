import { FaArrowRightLong } from "react-icons/fa6";
import Link from "next/link";

export default function Home() {
  const quickLinks = [
    {
      title: "Individual Patient",
      description: "Submit a single patient and get risk + XAI insights.",
      href: "/predict/individual",
    },
    {
      title: "Batch Prediction",
      description: "Upload a file and score multiple patients at once.",
      href: "/predict/batch",
    },
    {
      title: "Data Analysis",
      description: "Explore model behavior, EDA, and feature selection.",
      href: "/analyze/dashboard",
    },
  ];

  return (
    <div className="flex h-full items-center px-8 md:px-20">
      <div>
        <h1 className="text-6xl font-bold dark:text-white">
          Heart Failure Analytics
        </h1>
        <p className="text-xl mt-4 text-gray-500 max-w-3xl">
          Your go-to platform for AI-powered heart failure analysis and risk prediction.
          Combine machine learning with Explainable AI to gain accurate insights,
          understand model decisions, and explore patient data through preprocessing,
          EDA, and feature selection — all in one place.
        </p>
        <div className="mt-8">
          <div className="mt-4 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {quickLinks.map((item) => (
              <Link
                key={item.href}
                href={item.href}
                className="group rounded-2xl border border-gray-200/70 bg-white/70 p-4 shadow-sm transition hover:-translate-y-0.5 hover:border-indigo-300 hover:shadow-md dark:border-white/10 dark:bg-white/5"
              >
                <div className="flex items-center justify-between gap-4">
                  <div>
                    <p className="text-base font-semibold text-gray-900 dark:text-white">
                      {item.title}
                    </p>
                    <p className="mt-1 text-sm text-gray-500">
                      {item.description}
                    </p>
                  </div>
                  <span className="inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-indigo-50 text-indigo-600 transition group-hover:bg-indigo-100 dark:bg-white/10 dark:text-indigo-300">
                    <FaArrowRightLong className="text-sm" />
                  </span>
                </div>
              </Link>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}