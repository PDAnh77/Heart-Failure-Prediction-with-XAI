import { FaArrowRightLong } from "react-icons/fa6";
import Link from "next/link";

export default function Home() {
  return (
    <div className="flex h-full items-center px-8 md:px-20">
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
        <Link 
          href="/predict" 
          className="inline-flex gap-2 items-center mt-6 hover:cursor-pointer text-white bg-indigo-600 py-2 px-4 rounded-xl hover:bg-indigo-500 transition-colors"
        >
          Learn More
          <FaArrowRightLong />
        </Link>
        <div className="mt-8 max-w-2xl border-l-4 border-amber-400 pl-4 py-1">
          <p className="text-sm text-gray-500 dark:text-gray-400 italic leading-relaxed">
            <span className="font-semibold text-amber-600 dark:text-amber-500 not-italic">Disclaimer: </span>
            The information provided by this app is for educational purposes only and should not be considered medical advice. 
            Please consult a healthcare professional for medical guidance.
          </p>
        </div>
      </div>
    </div>
  );
}