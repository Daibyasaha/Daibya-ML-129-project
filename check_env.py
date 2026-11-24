import sys

packages = [
    "torch",
    "torchvision",
    "torchaudio",
    "scipy",
    "numpy",
    "pandas",
    "sklearn",
    "matplotlib",
    "seaborn",
    "PIL",
    "yaml",
    "pytest",
    "tqdm",
]

print(f"Python: {sys.version}")
missing = []
for pkg in packages:
    try:
        mod = __import__(pkg)
        ver = getattr(mod, "__version__", "unknown")
        print(f"  [OK] {pkg}: {ver}")
    except Exception as e:
        print(f"  [MISSING/ERROR] {pkg}: {e}")
        missing.append(pkg)

if missing:
    print(f"\nMissing packages: {missing}")
else:
    print("\nAll packages verified successfully!")
