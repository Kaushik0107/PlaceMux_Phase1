from config.config import CONFIG

print("Project Configuration")

for key, value in CONFIG.items():
    print(f"{key}: {value}")
    