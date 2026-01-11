"use client";

import toast, { Toast, ToastBar, Toaster } from "react-hot-toast";

export default function AppToaster() {
  return (
    <Toaster
      position="top-right"
      toastOptions={{ duration: 3000 }}
    >
      {(t: Toast) => (
        <ToastBar toast={t}>
          {({ icon, message }) => (
            <div className="flex items-center gap-3 w-full">
              {icon}
              <div className="flex-1">{message}</div>

              <button
                onClick={() => toast.dismiss(t.id)}
                className="text-sm text-slate-500 hover:text-slate-200 transition-all py-2 pr-2 cursor-pointer"
              >
                ✕
              </button>
            </div>
          )}
        </ToastBar>
      )}
    </Toaster>
  );
}
