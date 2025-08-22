import fs from 'fs'
import path from 'path'
/** @type {import('next').NextConfig} */
const nextConfig = {
  eslint: {
    ignoreDuringBuilds: true,
  },
  typescript: {
    ignoreBuildErrors: true,
  },
  images: {
    unoptimized: true,
  },
  webpack: (config, { isServer }) => {
    // Only on the client build, stub out "fs" so pptxgenjs won't pull it in
    if (!isServer) {
      config.resolve.fallback = {
        ...(config.resolve.fallback || {}),
        fs: false,
      }
    }
    return config
  },
  async rewrites() {
    // Prefer env var, else read from public/backend_port.txt
    let port = process.env.BACKEND_PORT
    if (!port) {
      try {
        const p = path.join(process.cwd(), 'public', 'backend_port.txt')
        port = fs.readFileSync(p, 'utf8').trim()
      } catch {
        // no-op; no rewrite if port unavailable
      }
    }

    if (!port) return []

    return [
      {
        source: '/api/:path*',
        destination: `http://localhost:${port}/api/:path*`,
      },
    ]
  },
}

export default nextConfig
