from datetime import datetime


class Logger:
    @staticmethod
    def info(title: str, **kwargs):
        print("\n" + "=" * 60)
        print(f"📄 {title}")
        print("-" * 60)

        for key, value in kwargs.items():
            print(f"{key:<20}: {value}")

        print("=" * 60)

    @staticmethod
    def success(message: str):
        print(f"✅ {message}")

    @staticmethod
    def warning(message: str):
        print(f"⚠️  {message}")

    @staticmethod
    def error(message: str):
        print(f"❌ {message}")
