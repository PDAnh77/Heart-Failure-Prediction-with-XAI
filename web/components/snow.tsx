'use client'
import { useEffect } from "react";

export default function Snow() {
    useEffect(() => {
        const canvas = document.getElementById("snow") as HTMLCanvasElement;
        const ctx = canvas.getContext("2d")!;

        const handleResize = () => {
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
        };
        window.addEventListener('resize', handleResize);
        handleResize();

        const flakes = Array.from({ length: 120 }, () => ({
            x: Math.random() * canvas.width,
            y: Math.random() * canvas.height,
            r: Math.random() * 2 + 1, // Kích thước
            d: Math.random() * 2 + 0.5, // Tốc độ rơi
        }));

        const draw = () => {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.fillStyle = "#dbe9fb";

            flakes.forEach(f => {
                ctx.beginPath();
                ctx.arc(f.x, f.y, f.r, 0, Math.PI * 2);
                ctx.fill();

                f.y += f.d;
                f.x += 0.5;
                if (f.y > canvas.height) {
                    f.y = 0;
                    f.x = Math.random() * canvas.width;
                }
                if (f.x > canvas.width) {
                    f.x = 0;
                }
            });

            requestAnimationFrame(draw);
        };

        draw();
        return () => window.removeEventListener('resize', handleResize);
    }, []);

    return <canvas id="snow" className="fixed inset-0 pointer-events-none z-40" />;
}