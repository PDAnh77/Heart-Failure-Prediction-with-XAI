import { MetadataRoute } from "next";

export default function robots(): MetadataRoute.Robots {
    return {
        rules: [
            {
                userAgent: "*",
                allow: "/",
                disallow: ["/setting", "/prediction-history/"]
            }
        ],
        sitemap: `${process.env.NEXT_APP_URL}/sitemap.xml`
    }
}