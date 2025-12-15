export default function SettingRow({ title, description, enabled, setEnabled }: { title: string; description: string; enabled: boolean; setEnabled: (value: boolean) => void }) {
    return (
        <div className="flex items-center justify-between py-4">
            <div className="flex flex-col pr-4">
                <span className="text-base font-medium text-gray-900 dark:text-white">
                    {title}
                </span>
                <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                    {description}
                </p>
            </div>
            <button
                onClick={() => setEnabled(!enabled)}
                className={`
                    relative inline-flex h-6 w-12 shrink-0 items-center rounded-full transition-colors duration-200 ease-in-out
                    ${enabled ? 'bg-indigo-600' : 'bg-gray-200 dark:bg-gray-700'}
                `}>
                <span
                    className={`
                        inline-block h-4 w-4 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out
                        ${enabled ? 'translate-x-7' : 'translate-x-1'}
                    `}
                />
            </button>
        </div>
    )
}