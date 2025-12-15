import { FaArrowRightLong } from "react-icons/fa6";
import Link from "next/link";

export default function Home() {
  return (
    <div className="flex h-full items-center px-20">
      <div>
        <h1 className="text-6xl font-bold dark:text-white">
          Heart Failure Predict
        </h1>
        <p className="text-xl mt-4 text-gray-500 max-w-3xl">
          Your go-to platform for AI-powered heart disease prediction, combining
          Artificial Intelligence with Explainable AI to deliver accurate risk
          assessments and transparent, interpretable insights for better
          healthcare decision-making.
        </p>
        <Link href="/predict" className="inline-flex gap-2 items-center mt-4 hover:cursor-pointer text-white bg-indigo-600 py-2 px-4 rounded-xl hover:bg-indigo-500">Learn More
          <FaArrowRightLong />
        </Link>
      </div>
    </div>
  );
}
