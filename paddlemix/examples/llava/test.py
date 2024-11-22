import sys
sys.path.append('/Users/little1d/Desktop/Code/PaddleMIX')
print(sys.path)

try:
    import paddlemix
    print("paddlemix imported successfully")
except ImportError as e:
    print(f"ImportError: {e}")