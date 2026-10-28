import os
from app import create_app
from waitress import serve

# Load production configuration
env = os.getenv("FLASK_ENV", "production")
app = create_app(env)

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 5000))
    threads = int(os.getenv("THREADS", 4))

    print("=" * 65)
    print("   STARTING PRODUCTION WAITRESS WSGI SERVER   ")
    print(f"   Listening on: http://{host}:{port}")
    print(f"   Worker Threads: {threads}")
    print(f"   Environment: {env}")
    print("=" * 65)

    serve(app, host=host, port=port, threads=threads)
